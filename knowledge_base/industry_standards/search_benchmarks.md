# Industry Benchmarks: Site Search Performance

## Overview

This document compiles industry benchmarks for site search performance across SaaS and e-commerce sectors.

## Key Metrics Benchmarks

### Search-to-Conversion Rate

| Industry | Poor | Average | Good | Excellent |
|----------|------|---------|------|-----------|
| E-commerce | < 1% | 1-3% | 3-5% | > 5% |
| SaaS | < 0.5% | 0.5-2% | 2-4% | > 4% |
| Media/Content | < 0.2% | 0.2-1% | 1-2% | > 2% |

**Source:** Algolia Search Benchmark Report 2025

### Click-Through Rate (CTR)

| Position | Average CTR | Top Performers |
|----------|-------------|----------------|
| Position 1 | 35-45% | 50-60% |
| Position 2 | 15-20% | 25-30% |
| Position 3 | 8-12% | 15-18% |
| Position 4+ | < 5% | < 10% |

**Key Insight:** First result position is critical. Improving result #1 relevance has 3x impact vs improving position #3.

### Zero-Result Rate

| Performance Level | Zero-Result Rate |
|-------------------|------------------|
| Poor | > 15% |
| Average | 8-15% |
| Good | 3-8% |
| Excellent | < 3% |

**Impact:** Each 1% reduction in zero-result rate correlates with 0.5% increase in conversion.

### Search Latency

| Latency (P95) | User Perception | Impact on Conversion |
|---------------|-----------------|----------------------|
| < 100ms | Instant | Baseline |
| 100-300ms | Fast | -2% conversion |
| 300-500ms | Noticeable | -5% conversion |
| > 500ms | Slow | -10% conversion |

**Source:** Google Search Quality Guidelines, adapted for site search

## Feature Adoption Benchmarks

### Autocomplete Usage

- 70% of users interact with autocomplete suggestions
- Users who use autocomplete convert 24% more often
- Optimal number of suggestions: 5-7

### Filters and Facets

- 40% of users apply at least one filter
- Filtered searches have 2x higher conversion rate
- Most used filters: Category, Price, Date

### Instant Answers

- Trigger rate for FAQ-type queries: 15-25%
- User satisfaction with instant answers: 4.1/5.0 average
- Reduction in support tickets: 10-20%

## Semantic Search vs Keyword Search

| Metric | Keyword Search | Semantic Search | Improvement |
|--------|----------------|-----------------|-------------|
| Zero-result rate | 12% | 4% | -67% |
| CTR on position 1 | 38% | 52% | +37% |
| Search-to-conversion | 2.1% | 3.8% | +81% |
| User satisfaction | 3.4/5 | 4.2/5 | +24% |

**Source:** Coveo AI Search Impact Study 2025

## Mobile vs Desktop

| Metric | Mobile | Desktop |
|--------|--------|---------|
| Searches per session | 1.2 | 1.8 |
| Average query length | 2.3 words | 3.1 words |
| CTR | 42% | 55% |
| Conversion rate | 1.8% | 3.2% |

**Key Insight:** Mobile users search less but convert at lower rates. Voice search and autocomplete are more critical on mobile.

## Recommendations for AI Search Features

Based on industry benchmarks, prioritize:

1. **Result relevance** - Position 1 CTR is the highest-leverage metric
2. **Zero-result handling** - Implement fallback strategies (suggestions, related content)
3. **Latency optimization** - Stay under 200ms P95
4. **Autocomplete quality** - 70% of users rely on it
5. **Instant answers** - High satisfaction, reduces support load

## Sources

- Algolia Search Benchmark Report 2025
- Coveo AI Search Impact Study 2025
- Google Search Quality Guidelines
- Baymard Institute E-commerce UX Research
- Forrester Wave: Cognitive Search 2025
