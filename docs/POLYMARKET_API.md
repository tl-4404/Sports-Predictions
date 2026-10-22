# Polymarket US API walkthrough

We are implementing Polymarket US market data one step at a time. The shared client
lives at `providers/polymarket.py`, so NBA and MLB can use it independently.

## Venue and authentication

Polymarket US has separate services:

| Base URL | Purpose | Credentials |
| --- | --- | --- |
| `https://gateway.polymarket.us` | Public events, markets, prices, order books | No API key |
| `https://api.polymarket.us` | Trading and account operations | Authenticated |

This client implements league metadata, a page of game events, and moneyline BBOs. The
international Gamma API, series IDs, and outcome-token approach from the initial
example do not describe the US integration.

## Step 1: request NBA league metadata

From the repository root:

```bash
python3 -m providers.polymarket
```

The script requires `requests`, already listed in `requirements.txt`.

```python
response = requests.get(
    "https://gateway.polymarket.us/v2/leagues/nba",
    timeout=20,
)
response.raise_for_status()
data = response.json()
league = data["league"]
```

- **GET** asks for data.
- **Base URL** identifies the service: `https://gateway.polymarket.us`.
- **Endpoint** identifies the resource: `/v2/leagues/{slug}`.
- **Path parameter** selects the league: `{slug}` becomes `nba`.
- **Slug** means a URL-friendly name. It differs from a numeric database ID.
- This request has no query parameters, body, or authentication headers.
- `timeout=20` limits network waiting, though it is not a total wall-clock deadline.
- `raise_for_status()` raises on HTTP failures such as 404 or 500.
- `.json()` parses the response body into Python dictionaries/lists and can fail
  separately if the body is invalid JSON.

The documented response is an object with a nested `league` object. The client
checks that structure and verifies the returned slug before returning the league.
The CLI pretty-prints that league with `json.dumps(..., indent=2)`.

Inspect `name`, `slug`, `isOperational`, and `activeSeriesId` in the output. This
response identifies the league; event discovery and executable prices come next.

## Step 2: fetch a page of NBA events

```bash
python3 -m providers.polymarket --events --limit 3
python3 -m providers.polymarket --events --limit 3 --offset 3
python3 -m providers.polymarket --events --limit 1 --raw
```

The first command prints a readable summary. The second requests the next page.
The third prints the complete API response for inspection. The returned events are
discovery data and are not yet filtered into eligible pregame predictions.

The core request in `fetch_league_events` is:

```python
response = requests.get(
    "https://gateway.polymarket.us/v2/leagues/nba/events",
    params={"limit": 3, "offset": 0, "type": "sport", "section": "general"},
    timeout=20,
)
response.raise_for_status()
data = response.json()
events = data["events"]
```

`params` is a dictionary of **query parameters**. Requests encodes it into the URL:

```text
/v2/leagues/nba/events?limit=3&offset=0&type=sport&section=general
```

The path selects NBA events; query parameters control which subset is returned.
`limit` requests a page size; `offset` specifies how many events to skip. `type=sport`
requests games rather than `futures`, and `section=general` selects the general
section rather than trending. This function fetches one page; it does not yet loop
through every page. An empty events list is a valid response. Live lists can change
between page requests, so offset pagination is not an immutable snapshot.

The response contains an `events` list and league/section metadata. Each event can
contain several `markets`, and each market contains its `marketSides`:

```text
Event: GS Warriors vs. LA Lakers
  Market: moneyline (who wins?)
    Side: Warriors, long=true
    Side: Lakers, long=false
```

In the live response inspected for this lesson, both sides referenced the market
slug `aec-nba-gs-lal-2026-10-21`. The moneyline type was
`SPORTS_MARKET_TYPE_MONEYLINE` and its `line` was null: no point handicap applies.
The event had `live=true`, so its prices must not be compared directly with a
pregame model. These values are an observed example and can change.

Use `marketSides` to inspect team identity and direction instead of deriving them
from the market title, list order, or home/away position. `long` identifies the
direction within the contract; it does not mean home versus away. The summary leaves
out display prices because they are not yet verified executable order-book quotes.

## Fetch moneylines and their current odds

```bash
python3 -m providers.polymarket --moneylines --limit 3
```

This performs two kinds of public request:

1. Fetch `/v2/leagues/nba/events` for one page of NBA games.
2. For each market whose `sportsMarketTypeV2` is
   `SPORTS_MARKET_TYPE_MONEYLINE`, fetch `/v1/markets/{slug}/bbo`.

`fetch_market_bbo()` returns the response's `marketData` object and verifies that
its `marketSlug` matches the requested slug. BBO means **best bid and offer**: the
highest price a buyer offers and the lowest price a seller accepts. Both quote the
single YES instrument in USD.

`moneyline_quotes()` maps the two market sides to entry costs:

| Side | Immediate entry | Cost per $1 potential payout |
| --- | --- | --- |
| `long=true` | Buy YES at the best ask | `bestAsk` |
| `long=false` | Sell YES at the best bid, equivalent to taking NO | `1 - bestBid` |

For a YES bid of $0.43 and ask of $0.44, the YES team costs $0.44 and the NO team
costs $0.57. The costs sum to $1.01 because of the spread. Taking `1 - ask` for the
NO team would incorrectly offer the buyer a better price than the book provides.

The returned `quotes` list contains event and team IDs, market slug, start time,
live flag, book state, quote basis, available `quoted_shares` when supplied by the
API, and a local UTC `observed_at`. That timestamp records local observation;
it does not prove the exchange quote is fresh. Missing `quoted_shares` means size
is unknown. The `odds` object contains:

```text
contract_price = c
decimal_odds = 1 / c
american_odds = +100 * (1 - c) / c      when c < 0.5
american_odds = -100 * c / (1 - c)      when c >= 0.5
```

Equivalent odds are rounded for display and exclude fees. Source prices are parsed
with `Decimal` to avoid arithmetic artifacts when taking complements.

Inactive/closed events or markets, nontradable sides, nonopen books, crossed books,
and missing/invalid/non-USD quotes are marked unavailable with `odds=null` and a
reason. Errors fetching a market appear in `errors`; the CLI exits nonzero if any
occur. An empty list is not substituted for a failed request.

This command reads one event page. Use `--offset` to inspect later pages. It may
include live games, with the live flag retained; pregame model filtering is a later
step. Quotes are fetched in sequence and can change between requests. The fetched
entry odds apply only to quoted available size and do not include trading fees.

## Next lessons

1. Inspect a market's rules; identify the game winner contract
   and which team the YES outcome represents.
2. Inspect full order-book depth, quote freshness, and fees for a chosen market.
3. Add pagination and pregame eligibility filters, store timestamped snapshots,
   and match event/team identity to NBA model outputs.
4. Calculate expected value using executable prices, applicable US fees, and the
   contract's settlement rules.

Our current NBA model predicts game winners. Spreads, totals, and player props
require their own probability models.

## EV example for a YES contract held to settlement

For a contract paying $1 when YES wins and $0 when NO wins, define `c` as the
purchase price and `p` as the model's probability of the exact YES outcome.
Ignoring fees:

```text
Expected profit per contract = p * (1 - c) + (1 - p) * (-c) = p - c
Expected return on cost = (p - c) / c
Equivalent decimal odds = 1 / c
```

A 60% forecast and a $0.55 purchase price give $0.05 expected profit per contract,
or about 9.09% return on cost, before fees. The best ask is the immediate YES buy
price only for the size available there; larger orders may cross several levels.
Displayed prices and last trades are insufficient to calculate executable EV.

## Official US references

- [API introduction](https://docs.polymarket.us/api-reference/introduction)
- [Get league by slug](https://docs.polymarket.us/api-reference/sports/get-league-by-slug)
- [Get events by league slug](https://docs.polymarket.us/api-reference/sports/get-events-by-league-slug)
- [Get market BBO](https://docs.polymarket.us/api-reference/markets/get-market-bbo)
- [US YES/NO trading model](https://docs.polymarket.us/concepts/orders)
- [Prices and market data](https://docs.polymarket.us/concepts/market-data)
- [Authentication](https://docs.polymarket.us/api-reference/authentication)
- [Fee schedule](https://docs.polymarket.us/fees)
