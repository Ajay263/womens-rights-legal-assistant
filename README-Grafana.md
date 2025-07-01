# Grafana Visualization for Chat Feedback

This project uses Grafana to visualize feedback and conversation data from the chat application.

## Setup

All components are configured in the docker-compose.yml file. To start the entire stack:

```bash
docker-compose up -d
```

## Accessing Grafana

Once the containers are running, access Grafana at:

```
http://localhost:3000
```

Default login credentials:
- Username: admin
- Password: admin

## Available Dashboards

The system comes pre-configured with a "Chat Feedback Dashboard" that includes:

1. **Conversations Over Time** - Line chart showing conversation volume trends
2. **Total Conversations** - Stat panel showing total conversation count
3. **Feedback Distribution (%)** - Pie chart showing percentage breakdown of positive (green) vs negative (red) feedback
4. **Users by Subscription Type** - Bar gauge showing user counts by subscription tier
5. **Response Time** - Graph showing LLM response time in seconds
6. **Answer Relevance Distribution** - Pie chart showing the distribution of answer relevance as judged by the LLM
7. **Token Usage Over Time** - Stacked area chart showing prompt and completion tokens
8. **Total API Cost** - Stat panel showing the total cost of Groq API usage
9. **Recent Conversations with Metrics** - Table showing the most recent conversations with all metrics

## LLM Metrics Monitoring

The dashboard provides comprehensive monitoring of LLM performance metrics:

### Response Time
- Tracks how long it takes to get a response from the LLM
- Shows min, max, and average response times

### Answer Relevance
- Uses an LLM as a judge to evaluate the relevance of each answer
- Categorizes answers as "Relevant", "Partly Relevant", or "Not Relevant"
- Shows the distribution as a percentage

### Token Usage
- Tracks prompt tokens and completion tokens separately
- Shows token usage trends over time
- Helps optimize costs and performance

### API Costs
- Calculates and displays the total cost of Groq API usage
- Based on Groq's pricing model for LLama 3 70B

## Customizing Dashboards

You can modify the existing dashboard or create new ones through the Grafana UI. Any changes made through the UI will be persistent across container restarts because of the grafana_data volume.

## Adding Custom Metrics

To add custom metrics:

1. Log into Grafana
2. Navigate to the dashboard you want to modify
3. Click "Add Panel"
4. Select visualization type
5. Write a SQL query to extract the data you need
6. Save the dashboard

## Troubleshooting

If you don't see data in Grafana:

1. Check that the PostgreSQL container is running: `docker-compose ps`
2. Verify the database connection in Grafana (Configuration → Data Sources)
3. Check the Grafana logs: `docker-compose logs grafana`
4. Ensure the application is actually saving data to the PostgreSQL database 