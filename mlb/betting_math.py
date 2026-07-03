"""Shared betting math: odds conversion, de-vigging, Kelly staking."""


def american_to_prob(odds):
    """Implied probability including vig."""
    if odds is None:
        return None
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / -odds


def devig_two_way(over_odds, under_odds):
    """Multiplicative de-vig of a two-way market.

    Returns (fair_p_over, fair_p_under) or (None, None) if either side missing.
    """
    p_o = american_to_prob(over_odds)
    p_u = american_to_prob(under_odds)
    if p_o is None or p_u is None or (p_o + p_u) <= 0:
        return None, None
    total = p_o + p_u
    return p_o / total, p_u / total


def consensus_p_over(price_pairs):
    """Average de-vigged over-probability across books.

    price_pairs: iterable of (over_odds, under_odds). Books missing a side are
    skipped. Returns None when nothing usable.
    """
    fair = []
    for over_odds, under_odds in price_pairs:
        p, _ = devig_two_way(over_odds, under_odds)
        if p is not None:
            fair.append(p)
    return sum(fair) / len(fair) if fair else None


def kelly_fraction(p_win, american_odds, multiplier=0.25, cap=0.02):
    """Fractional Kelly stake as a share of bankroll.

    multiplier: fraction of full Kelly (0.25 = quarter Kelly, standard for
    model uncertainty). cap: hard ceiling per bet. Returns 0 when no edge.
    """
    if p_win is None or american_odds is None:
        return 0.0
    b = american_to_decimal(american_odds) - 1.0  # net odds
    if b <= 0:
        return 0.0
    full = (p_win * b - (1.0 - p_win)) / b
    return max(0.0, min(full * multiplier, cap))
