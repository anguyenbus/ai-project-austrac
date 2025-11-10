# PostgreSQL AnalyticsConnector Walkthrough

This directory contains examples demonstrating how to use `AnalyticsConnector` with PostgreSQL databases.

## Prerequisites

Before running the examples, you need to set up the Northwind PostgreSQL database.

### 1. Clone the Northwind Database Repository

The Northwind database is located in the `northwind/` directory at the project root. If you don't have it cloned:

```bash
# From project root
cd /home/user/projects/instantinsight-db
ls northwind/  # Should see docker-compose.yml, northwind.sql, etc.
```

The Northwind repository is already included in this project at the root level.

### 2. Start the Northwind Docker Container

The Northwind database runs on **port 5555** to avoid conflicts with the main RAG PostgreSQL database (port 5432).

```bash
# Navigate to the northwind directory
cd northwind/

# Start the PostgreSQL container with Docker Compose
docker-compose up -d

# Verify the container is running
docker-compose ps
```

**Expected output:**

```
NAME                    IMAGE         STATUS          PORTS
northwind-postgres-1    postgres:16   Up 10 seconds   0.0.0.0:5555->5432/tcp
```

### 3. Verify the Database is Ready

Wait a few seconds for the database to initialize, then test the connection:

```bash
# Test connection using psql
docker exec -it northwind-postgres-1 psql -U postgres -d northwind -c "SELECT COUNT(*) FROM customers;"
```

**Expected output:**

```
 count 
-------
    91
(1 row)
```

## Running the Walkthrough

Once the Northwind database is running, you can execute the walkthrough script:

```bash
# From project root
cd /home/user/projects/instantinsight-db

# Run the walkthrough
uv run python examples/posgresql/ibis_postgres_walkthrough.py
```

### What the Walkthrough Covers

The [`ibis_postgres_walkthrough.py`](./ibis_postgres_walkthrough.py:1) script demonstrates:

1. **Connection Setup** - Connecting to PostgreSQL via `AnalyticsConnector`
2. **Database Discovery** - Listing databases, schemas, and tables
3. **Simple Queries** - Basic SELECT statements
4. **JOINs** - Multi-table queries
5. **Aggregations** - COUNT, SUM, AVG, GROUP BY
6. **Complex Queries** - Subqueries and CTEs
7. **Schema Introspection** - Using `SchemaIntrospector`
8. **DataFrame Operations** - Working with pandas DataFrames
9. **Business Queries** - Practical examples (sales trends, customer analytics)
10. **Context Managers** - Proper resource cleanup

## Connection Details

**Northwind Database Configuration:**

- Host: `localhost`
- Port: `5555` (not 5432!)
- Database: `northwind`
- Username: `postgres`
- Password: `postgres`

**Connection String:**

```python
connection_string = "postgres://postgres:postgres@localhost:5555/northwind"
```

## Database Schema

The Northwind database contains 14 tables:

- `customers` - Customer information
- `orders` - Order records
- `order_details` - Order line items
- `products` - Product catalog
- `categories` - Product categories
- `suppliers` - Supplier information
- `employees` - Employee records
- `employee_territories` - Employee territory assignments
- `territories` - Territory definitions
- `regions` - Geographic regions
- `shippers` - Shipping companies
- `customer_demographics` - Customer demographic data
- `customer_customer_demo` - Customer-demographic mapping
- `us_states` - US state information

**Total Records:** ~3,300 across all tables

## Stopping the Database

When you're done with the examples:

```bash
# Navigate to northwind directory
cd northwind/

# Stop the container (data persists)
docker-compose down

# OR: Stop and remove all data
docker-compose down -v
```

## Troubleshooting

### Port Conflict

If you see "port 5555 already in use":

```bash
# Check what's using the port
lsof -i :5555

# Kill the process or change the port in northwind/docker-compose.yml
```

### Connection Refused

If the walkthrough can't connect:

```bash
# Verify container is running
docker-compose ps

# Check container logs
docker-compose logs

# Restart the container
docker-compose restart
```

### Database Not Initialized

If tables are missing:

```bash
# Recreate the database
docker-compose down -v
docker-compose up -d

# Wait 10 seconds for initialization
sleep 10
```

## Next Steps

After completing the walkthrough:

1. **Try Different Queries** - Modify the walkthrough script
2. **Connect Your Own Database** - Use `AnalyticsConnector` with other PostgreSQL instances
3. **RAG Integration** - Use `SchemaVectorizer` for text-to-SQL applications

## Additional Resources

- **AnalyticsConnector Documentation**: [`src/connectors/analytics_backend.py`](../../src/connectors/analytics_backend.py:1)
- **SchemaIntrospector Documentation**: [`src/utils/schema_introspector.py`](../../src/utils/schema_introspector.py:1)
- **Northwind ER Diagram**: [`northwind/ER.png`](../../northwind/ER.png:1)
- **Test Suite**: [`tests/integration/`](../../tests/integration/test_northwind_connectivity.py:1)

## Quick Start Summary

```bash
# 1. Start Northwind database
cd northwind && docker-compose up -d

# 2. Run walkthrough (from project root)
cd .. && uv run python examples/posgresql/ibis_postgres_walkthrough.py

# 3. Stop database when done
cd northwind && docker-compose down
```

---

**Last Updated:** 2025-10-26
**Python Version:** 3.13+
**Dependencies:** `uv`, `docker`, `docker-compose`
