# Search Analytics Dashboard Specification

## Purpose

This document defines the analytics requirements for monitoring AI Search performance and user behavior.

## Dashboard Sections

### 1. Search Volume & Trends

**Metrics:**
- Total searches per day/week/month
- Unique users searching
- Searches per session
- Peak search hours

**Visualizations:**
- Time series of search volume
- Hour-of-day heatmap
- Day-of-week distribution

### 2. Search Quality

**Metrics:**
- Zero-result rate: Percentage of searches returning no results
- Click-through rate (CTR): Percentage of searches with at least one click
- Mean Reciprocal Rank (MRR): Average position of first clicked result
- Abandonment rate: Searches with no interaction

**Targets:**
| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Zero-result rate | < 5% | > 10% |
| CTR | > 60% | < 40% |
| MRR | > 0.7 | < 0.5 |
| Abandonment rate | < 20% | > 35% |

### 3. Conversion Funnel

**Stages:**
1. Search initiated
2. Results viewed
3. Result clicked
4. Conversion (signup, purchase, etc.)

**Metrics:**
- Conversion rate by search query category
- Drop-off rate between stages
- Time to conversion from search

### 4. Query Analysis

**Metrics:**
- Top 100 queries by volume
- Top queries with zero results
- Top queries with high abandonment
- Query length distribution

**Segmentation:**
- By user type (new vs returning)
- By device (mobile vs desktop)
- By source page

### 5. Instant Answers Performance

**Metrics:**
- Instant answer trigger rate
- Helpfulness rating distribution
- Click-through after instant answer

## Data Sources

### Primary Tables

**`search_events`** - Event-level search data
- `event_id` (PK)
- `user_id`
- `session_id`
- `query`
- `result_count`
- `latency_ms`
- `timestamp`

**`search_clicks`** - Click-level data
- `click_id` (PK)
- `search_event_id` (FK)
- `result_position`
- `result_url`
- `timestamp`

**`search_conversions`** - Conversion attribution
- `conversion_id` (PK)
- `search_event_id` (FK)
- `conversion_type`
- `conversion_value`
- `timestamp`

## Refresh Frequency

| Dashboard Section | Refresh Rate |
|-------------------|--------------|
| Real-time metrics | 1 minute |
| Daily aggregates | Hourly |
| Weekly trends | Daily |

## Access Control

- **Product team:** Full access
- **Engineering:** Full access
- **Marketing:** Read-only, no PII
- **Executive:** Summary dashboard only
