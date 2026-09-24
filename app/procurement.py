"""Replenishment procurement planning for one saved recipe version.

Given the version's solve result (``allocations`` / ``total_volume_ml``),
the expanded irrigation schedule (reusing :func:`irrigation.build_plan`,
i.e. the same DST-aware expansion, cross-midnight handling and ≤ 1 min
merging as the forecast), the current stock of every concentrate and a
list of supplier *quotes* (whole packs of one concentrate with a fixed
pack size, per-pack price, local arrival date and maximum purchasable
pack count), this module finds the minimum-total-price whole-pack
purchase combination such that stock never goes negative:

* every actual (post-merge) irrigation batch of water ``W`` draws
  ``required_i = allocation_i / total_volume_ml * W`` mL from bottle
  *i* — merged and cross-midnight batches are deducted exactly once,
  at their start instant, and attributed to their local start date;
* a quote arriving on local date ``D`` is inbound **before the first
  batch whose local start date is ``D``**; a quote arriving after a
  batch cannot serve it (this is the "quote timing" constraint);
* the number of purchased packs of every quote is a non-negative
  integer not exceeding its supply cap ``max_packs``.

Feasibility / earliest-gap detection
------------------------------------

Purchasing is separable per concentrate, so feasibility is checked with
a greedy *maximum-supply* pass: at every batch each bottle receives all
volume that **could** ever arrive by then (``max_packs * pack_size`` of
every quote with ``arrival_date <= batch local date``).  The first batch
where even that supply cannot cover the requirement is the earliest
unavoidable gap (late quotes and tight supply caps alike).  In that case
no executable order is fabricated: every quote reports ``packs = 0`` and
the first gap batch + short concentrates are returned.

Optimisation
------------

When the greedy pass succeeds, a pure integer linear program (HiGHS via
``scipy.optimize.milp``) decides the pack counts:

    variables  x_j in {0, ..., quote_j.max_packs}
    s.t.  stock_i + sum_{j of i, arrival_j <= date_t} size_j * x_j
              >= sum_{s<=t} required_{i,s}          for every batch t
    stage 1: minimise sum_j price_j * x_j            (最低总价)
    stage 2: among minimum-price solutions,
             minimise sum_j x_j                      (同价总包数最少)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .irrigation import build_plan

# Results are reported to 4 decimal places (mL / money units).
_ROUND = 4
# mL tolerance shared with the forecast: exact stock exhaustion over
# several batches must not read as a phantom negative / shortage.
_EPS = 1e-7
# Slack for re-fixing the stage-1 minimum price in the tie-break stage so
# the second MILP cannot lose the optimum to floating-point noise.
_PRICE_SLACK_REL = 1e-9
_PRICE_SLACK_ABS = 1e-7


class ProcurementError(ValueError):
    """Stock/quote list does not match the recipe version's concentrates."""


@dataclass
class _Quote:
    """One supplier offer: whole packs of one concentrate."""

    index: int
    name: str
    size: float
    price: float
    arrival: date
    max_packs: int


def _r(x: float) -> float:
    """Round for output, normalizing -0.0 produced by float arithmetic."""
    v = round(float(x), _ROUND)
    return 0.0 if v == 0 else v


def build_procurement_plan(
    *,
    recipe_id: int,
    version: int,
    solve_result: dict[str, Any],
    tz: ZoneInfo,
    start_date: date,
    end_date: date,
    sessions: list[tuple],
    stock: dict[str, float],
    quotes: list[tuple[str, float, float, date, int]],
    suspended_dates: set | None = None,
) -> dict[str, Any]:
    """Compute the minimum-cost whole-pack procurement plan.

    ``stock`` maps every concentrate name used by the version result to
    its current on-hand mL.  ``quotes`` is an ordered list of
    ``(concentrate, pack_size_ml, price_per_pack, arrival_date,
    max_packs)``.  Raises :class:`ProcurementError` when stock or quote
    names do not match the version, or the persisted volume is invalid.
    """
    allocations = solve_result.get("allocations") or []
    names = [str(a.get("concentrate")) for a in allocations]
    total_volume_ml = float(solve_result["total_volume_ml"])
    if total_volume_ml <= 0:
        raise ProcurementError(
            "配方版本的目标总体积必须为正，无法按比例折算用液需求。"
        )

    unknown = sorted(set(stock) - set(names))
    if unknown:
        raise ProcurementError(
            f"库存中存在配方版本未使用的未知浓缩液：{ '、'.join(unknown) }；"
            f"该版本浓缩液为：{ '、'.join(names) }。"
        )
    missing_stock = [n for n in names if n not in stock]
    if missing_stock:
        raise ProcurementError(
            f"库存缺少该配方版本使用的浓缩液：{ '、'.join(missing_stock) }；"
            "请为每一种浓缩液提供当前库存。"
        )

    quote_list: list[_Quote] = []
    for i, (qname, size, price, arrival, max_packs) in enumerate(
        quotes, start=1
    ):
        if qname not in names:
            raise ProcurementError(
                f"报价 {i} 的浓缩液 '{qname}' 不在该配方版本中；"
                f"该版本浓缩液为：{ '、'.join(names) }。"
            )
        quote_list.append(
            _Quote(
                index=i,
                name=qname,
                size=float(size),
                price=float(price),
                arrival=arrival,
                max_packs=int(max_packs),
            )
        )

    # mL of concentrate per mL of irrigation water, per bottle.
    ratio = {
        n: float(a.get("volume_ml", 0.0)) / total_volume_ml
        for n, a in zip(names, allocations)
    }
    name_index = {n: i for i, n in enumerate(names)}
    initial_stock = {n: float(stock[n]) for n in names}

    # Same expansion as the consumption forecast (DST / cross-midnight /
    # ≤1-minute merge; no device-capacity input here).
    plan = build_plan(
        tz=tz,
        start_date=start_date,
        end_date=end_date,
        sessions=sessions,
        daily_capacity_ml=None,
        suspended_dates=suspended_dates,
    )
    raw_batches = plan["batches"]

    # Per-batch deduction (once per merged batch, at its start instant).
    batch_dates: list[date] = []
    demands: list[dict[str, float]] = []
    cumulative = {n: 0.0 for n in names}
    cumulative_at: list[dict[str, float]] = []
    for b in raw_batches:
        water = float(b["water_ml"])
        d = date.fromisoformat(b["local_start"][:10])
        need = {n: ratio[n] * water for n in names}
        for n in names:
            cumulative[n] += need[n]
        batch_dates.append(d)
        demands.append(need)
        cumulative_at.append(dict(cumulative))

    total_required = {
        n: sum(ratio[n] * float(b["water_ml"]) for b in raw_batches)
        for n in names
    }

    base_meta = {
        "recipe_id": recipe_id,
        "version": version,
        "timezone": plan["timezone"],
        "start_date": plan["start_date"],
        "end_date": plan["end_date"],
        "days": plan["days"],
        "suspended_dates": plan["suspended_dates"],
        "batch_count_before_merge": plan["batch_count_before_merge"],
        "batch_count_after_merge": plan["batch_count_after_merge"],
        "planned_water_ml": _r(plan["planned_water_ml"]),
    }

    feasible_version = bool(solve_result.get("feasible"))
    warning: Optional[str] = None
    if not feasible_version:
        warning = (
            "该配方版本的求解结果在容差范围内不可行（feasible=false），"
            "用液需求按其折中方案的 allocations 折算，采购方案仅供参考。"
        )

    # ---- 1. maximum-supply greedy pass: earliest unavoidable gap -------- #
    # (arrival_date, name index, mL that can ever arrive under the cap)
    max_events = sorted(
        (
            q.arrival,
            name_index[q.name],
            q.size * q.max_packs,
        )
        for q in quote_list
    )
    remaining_max = dict(initial_stock)
    event_ptr = 0
    first_gap: Optional[dict[str, Any]] = None

    for t, b in enumerate(raw_batches):
        cur_date = batch_dates[t]
        while event_ptr < len(max_events) and (
            max_events[event_ptr][0] <= cur_date
        ):
            _, ni, vol = max_events[event_ptr]
            remaining_max[names[ni]] += vol
            event_ptr += 1

        liquids_gap = []
        batch_short = False
        for n in names:
            required = demands[t][n]
            available = remaining_max[n]
            if required <= available + _EPS:
                shortfall = 0.0
                remaining_max[n] = available - required
            else:
                shortfall = required - max(available, 0.0)
                batch_short = True
                remaining_max[n] = 0.0
            liquids_gap.append(
                {
                    "concentrate": n,
                    "required_ml": _r(required),
                    "available_ml": _r(available),
                    "shortfall_ml": _r(shortfall),
                }
            )
        if batch_short and first_gap is None:
            first_gap = {
                "index": t + 1,
                "start_utc": b["start_utc"],
                "local_start": b["local_start"],
                "local_end": b["local_end"],
                "attributed_date": b["local_start"][:10],
                "water_ml": _r(float(b["water_ml"])),
                "merged": b["merged"],
                "batch_count_before_merge": b["batch_count_before_merge"],
                "crosses_midnight": b["crosses_midnight"],
                "liquids": liquids_gap,
            }
            break

    def quote_echo(packs: dict[int, int]) -> list[dict[str, Any]]:
        out = []
        for q in quote_list:
            k = int(packs.get(q.index, 0))
            out.append(
                {
                    "index": q.index,
                    "concentrate": q.name,
                    "pack_size_ml": _r(q.size),
                    "price": _r(q.price),
                    "arrival_date": q.arrival.isoformat(),
                    "max_packs": q.max_packs,
                    "packs": k,
                    "volume_ml": _r(k * q.size),
                    "cost": _r(k * q.price),
                }
            )
        return out

    def liquid_summary(
        purchased: dict[str, float], final: dict[str, float]
    ) -> list[dict[str, Any]]:
        return [
            {
                "concentrate": n,
                "initial_stock_ml": _r(initial_stock[n]),
                "mix_ml_per_l": _r(ratio[n] * 1000.0),
                "total_required_ml": _r(total_required[n]),
                "total_purchased_ml": _r(purchased.get(n, 0.0)),
                "final_remaining_ml": _r(final.get(n, initial_stock[n])),
            }
            for n in names
        ]

    if first_gap is not None:
        short_names = [
            l["concentrate"]
            for l in first_gap["liquids"]
            if l["shortfall_ml"] > 0
        ]
        first_gap["message"] = (
            f"最早缺口出现在第 {first_gap['index']} 批"
            f"（本地 {first_gap['local_start']} 开始）：浓缩液 "
            f"{ '、'.join(short_names) } 在该批扣减后库存将为负；"
            "现有报价的到货时点或最多可购包数无法补足缺口，"
            "未生成可执行采购单。"
        )
        return {
            **base_meta,
            "recipe_feasible": feasible_version,
            "warning": warning,
            "feasible": False,
            "optimal": False,
            "quotes": quote_echo({}),
            "total_packs": 0,
            "total_price": 0.0,
            "liquids": liquid_summary({}, dict(initial_stock)),
            "batches": [],
            "first_gap": first_gap,
            "message": (
                "在现有库存与报价下无法保证每批扣减后库存非负："
                "请增加供应上限或安排更早到货的报价。"
            ),
        }

    # ---- 2. MILP: minimum total price, then minimum total packs --------- #
    packs_opt, optimal = _optimise_packs(
        quote_list=quote_list,
        names=names,
        name_index=name_index,
        initial_stock=initial_stock,
        batch_dates=batch_dates,
        cumulative_at=cumulative_at,
    )

    # ---- 3. forward simulation with the purchased packs ----------------- #
    # (arrival_date, name index, packs, mL) — pointer-released at the
    # first batch on/after each arrival date, so goods arriving on a day
    # with no batch (e.g. suspended day) enter stock at the next batch.
    inbound_events = sorted(
        (
            q.arrival,
            name_index[q.name],
            packs_opt[q.index],
            packs_opt[q.index] * q.size,
            q.index,
        )
        for q in quote_list
        if packs_opt[q.index] > 0
    )

    remaining = dict(initial_stock)
    ptr = 0
    batches_out: list[dict[str, Any]] = []
    purchased_total = {n: 0.0 for n in names}
    for t, b in enumerate(raw_batches):
        cur_date = batch_dates[t]
        inbound_agg: dict[str, dict[str, float]] = {}
        while ptr < len(inbound_events) and (
            inbound_events[ptr][0] <= cur_date
        ):
            _, ni, k, vol, _ = inbound_events[ptr]
            n = names[ni]
            remaining[n] += vol
            purchased_total[n] += vol
            agg = inbound_agg.setdefault(n, {"packs": 0, "volume_ml": 0.0})
            agg["packs"] += int(k)
            agg["volume_ml"] += vol
            ptr += 1

        liquids_out = []
        for n in names:
            beginning = remaining[n]
            after = beginning - demands[t][n]
            # The MILP guarantees this; the epsilon only covers FP noise.
            if after < 0 and after >= -_EPS:
                after = 0.0
            remaining[n] = after
            liquids_out.append(
                {
                    "concentrate": n,
                    "required_ml": _r(demands[t][n]),
                    "beginning_ml": _r(beginning),
                    "remaining_ml": _r(after),
                }
            )

        batches_out.append(
            {
                "index": t + 1,
                "start_utc": b["start_utc"],
                "local_start": b["local_start"],
                "local_end": b["local_end"],
                "attributed_date": b["local_start"][:10],
                "water_ml": _r(float(b["water_ml"])),
                "merged": b["merged"],
                "batch_count_before_merge": b["batch_count_before_merge"],
                "crosses_midnight": b["crosses_midnight"],
                "inbound": [
                    {
                        "concentrate": n,
                        "packs": int(v["packs"]),
                        "volume_ml": _r(v["volume_ml"]),
                    }
                    for n, v in inbound_agg.items()
                ],
                "liquids": liquids_out,
            }
        )

    quote_rows = quote_echo(packs_opt)
    total_packs = sum(q["packs"] for q in quote_rows)
    total_price = _r(sum(q["cost"] for q in quote_rows))

    return {
        **base_meta,
        "recipe_feasible": feasible_version,
        "warning": warning,
        "feasible": True,
        "optimal": optimal,
        "quotes": quote_rows,
        "total_packs": total_packs,
        "total_price": total_price,
        "liquids": liquid_summary(purchased_total, remaining),
        "batches": batches_out,
        "first_gap": None,
        "message": (
            f"采购方案求解成功：共采购 {total_packs} 包，"
            f"总价 {total_price:g}；各批次扣减后库存均非负。"
        ),
    }


# --------------------------------------------------------------------------- #
# MILP
# --------------------------------------------------------------------------- #
def _optimise_packs(
    *,
    quote_list: list[_Quote],
    names: list[str],
    name_index: dict[str, int],
    initial_stock: dict[str, float],
    batch_dates: list[date],
    cumulative_at: list[dict[str, float]],
) -> tuple[dict[int, int], bool]:
    """Two-stage integer program over whole-pack purchase counts.

    Returns ``(packs_by_quote_index, optimal_flag)``.  The greedy
    maximum-supply pass has already proved feasibility, so a failure
    here is purely numerical/defensive; the maximum-supply counts are
    returned in that case (they are feasible, albeit not cheapest).
    """
    qcount = len(quote_list)
    fallback = {q.index: q.max_packs for q in quote_list}
    zero_packs = {q.index: 0 for q in quote_list}
    if qcount == 0 or not batch_dates:
        return zero_packs, True

    sizes = np.array([q.size for q in quote_list], dtype=float)
    prices = np.array([q.price for q in quote_list], dtype=float)
    caps = np.array([q.max_packs for q in quote_list], dtype=float)
    q_names = np.array([name_index[q.name] for q in quote_list], dtype=int)
    arrivals = [q.arrival for q in quote_list]

    # One cumulative-coverage row per (batch, concentrate):
    #   sum_{j of i, arrival_j <= date_t} size_j * x_j
    #       >= cumulative requirement of i through batch t - stock_i
    rows: list[np.ndarray] = []
    lowers: list[float] = []
    needs_purchase = False
    # Which quotes serve concentrate i.
    quotes_of: list[np.ndarray] = [
        np.flatnonzero(q_names == ni) for ni in range(len(names))
    ]
    for cur_date, cum in zip(batch_dates, cumulative_at):
        available_mask = np.array(
            [a <= cur_date for a in arrivals], dtype=bool
        )
        for ni, n in enumerate(names):
            rhs = cum[n] - initial_stock[n]
            if rhs <= _EPS:
                continue
            needs_purchase = True
            row = np.zeros(qcount)
            idx = quotes_of[ni][available_mask[quotes_of[ni]]]
            row[idx] = sizes[idx]
            rows.append(row)
            # Slight downward slack so exact-exhaustion solutions are not
            # lost to floating-point accumulation (matches forecast _EPS).
            lowers.append(rhs - _EPS)

    if not needs_purchase:
        # Initial stock alone covers every batch: buying nothing is the
        # unique optimum for both objectives (all prices are positive).
        return zero_packs, True

    constraints = LinearConstraint(
        np.vstack(rows), np.array(lowers), np.inf
    )
    bounds = Bounds(np.zeros(qcount), caps)
    integrality = np.ones(qcount)

    # Stage 1: minimum total price.
    res1 = milp(
        c=prices,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    if res1 is None or res1.x is None:
        return fallback, False

    price_star = float(res1.fun)

    # Stage 2: keep the minimum price, minimise total pack count.
    # Coverage rows keep their one-sided lower bounds; the added price row
    # is a one-sided upper bound (price · x <= stage-1 optimum + slack).
    price_cap = price_star * (1.0 + _PRICE_SLACK_REL) + _PRICE_SLACK_ABS
    rows2 = rows + [prices.copy()]
    lb2 = np.array(lowers + [-np.inf])
    ub2 = np.array([np.inf] * len(rows) + [price_cap])
    constraints2 = LinearConstraint(np.vstack(rows2), lb2, ub2)

    res2 = milp(
        c=np.ones(qcount),
        integrality=integrality,
        bounds=bounds,
        constraints=constraints2,
    )
    res = res2 if (res2 is not None and res2.x is not None) else res1
    optimal = bool(res.status == 0)

    packs = np.rint(res.x).astype(int)
    packs = np.clip(packs, 0, caps.astype(int))
    packs_by_index = {
        quote_list[j].index: int(packs[j]) for j in range(qcount)
    }

    # Defensive: verify the rounded solution still satisfies every coverage
    # row exactly.  Fall back to the (feasible) max-supply counts rather
    # than ever emitting an order that could drive stock negative.
    x = packs.astype(float)
    coverage_mat = np.vstack(rows)
    if np.any(coverage_mat @ x < np.array(lowers) - 1e-6):
        return fallback, False
    return packs_by_index, optimal
