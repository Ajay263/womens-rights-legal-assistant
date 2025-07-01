# Adding Relevancy Gauge Panels to Grafana

To visualize the relevancy metrics with gauges, follow these instructions to add them manually in the Grafana UI:

## 1. Add Relevant Answers Gauge

1. Open your Grafana dashboard
2. Click "Add panel" (+ icon)
3. Select "Gauge" visualization
4. Configure the query:

```sql
SELECT
  NOW() as time,
  (COUNT(CASE WHEN relevance = 'RELEVANT' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0)) as value
FROM conversation
WHERE relevance IS NOT NULL
```

5. Set the following options:
   - Title: "Relevant Answers %"
   - Unit: "Percent (0-100)"
   - Min: 0
   - Max: 100
   - Thresholds:
     - 0-50: Red
     - 50-75: Yellow
     - 75-100: Green

## 2. Add Partly Relevant Answers Gauge

1. Click "Add panel" again
2. Select "Gauge" visualization
3. Configure the query:

```sql
SELECT
  NOW() as time,
  (COUNT(CASE WHEN relevance = 'PARTLY_RELEVANT' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0)) as value
FROM conversation
WHERE relevance IS NOT NULL
```

4. Set the following options:
   - Title: "Partly Relevant Answers %"
   - Unit: "Percent (0-100)"
   - Min: 0
   - Max: 100
   - Thresholds:
     - 0-25: Green
     - 25-50: Yellow
     - 50-100: Red

## 3. Add Non-Relevant Answers Gauge

1. Click "Add panel" again
2. Select "Gauge" visualization
3. Configure the query:

```sql
SELECT
  NOW() as time,
  (COUNT(CASE WHEN relevance = 'NON_RELEVANT' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0)) as value
FROM conversation
WHERE relevance IS NOT NULL
```

4. Set the following options:
   - Title: "Non-Relevant Answers %"
   - Unit: "Percent (0-100)"
   - Min: 0
   - Max: 100
   - Thresholds:
     - 0-10: Green
     - 10-25: Yellow
     - 25-100: Red

## 4. Arrange Panels

Arrange the three gauge panels in a row for a clean visualization of your relevancy metrics. You can drag and resize them as needed.

## 5. Update Feedback Pie Chart Colors

For the feedback pie chart, update the colors:
1. Edit the "Feedback Distribution (%)" panel
2. Go to "Field" tab
3. Add field overrides:
   - For "Positive Feedback": Set color to bright green (#73BF69)
   - For "Negative Feedback": Set color to bright red (#F2495C)

These gauges will give you a clear visual indication of your LLM's performance in terms of relevancy. 