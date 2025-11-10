"""
AnalyticsConnector PostgreSQL Walkthrough.

A step-by-step guide demonstrating how to use AnalyticsConnector to connect to
PostgreSQL and execute queries using the Northwind database.

This script can be run section by section in an interactive Python session
or as a complete script.

Date: 2025-10-26
"""

from rich.console import Console
from rich.table import Table

from src.connectors.analytics_backend import AnalyticsConnector
from src.utils.schema_introspector import SchemaIntrospector

# ============================================================================
# SECTION 1: Setup and Imports
# ============================================================================
print("=" * 80)
print("SECTION 1: Setup and Imports")
print("=" * 80)

console = Console()

# ============================================================================
# SECTION 2: Connection Setup
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Connecting to PostgreSQL (Northwind Database)")
print("=" * 80)

# Define connection parameters
HOST = "localhost"
PORT = "5555"  # Northwind runs on 5555 to avoid conflict with RAG DB (5432)
DATABASE = "northwind"
USER = "postgres"
PASSWORD = "postgres"

# Construct connection string
# Format: postgres://user:password@host:port/database
connection_string = f"postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

console.print(f"\n[bold cyan]Connection String:[/] {connection_string}", style="dim")

# Create AnalyticsConnector instance
backend = AnalyticsConnector(connection_string)

console.print(f"✓ [green]Connected to {backend.backend_type} backend[/]")
console.print(f"✓ [green]Backend type: {backend.backend_type}[/]")

# ============================================================================
# SECTION 3: Discovering Databases and Tables
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Discovering Databases and Tables")
print("=" * 80)

# List available databases/schemas
databases = backend.list_databases()
console.print(f"\n[bold]Available schemas:[/] {len(databases)}")
for db in databases:
    console.print(f"  • {db}")

# List tables in the Northwind database
# NOTE: For PostgreSQL, pass None to list tables in current database
tables = backend.list_tables(database=None)
console.print(f"\n[bold]Tables in Northwind:[/] {len(tables)}")
for table in sorted(tables):
    console.print(f"  • {table}")

# ============================================================================
# SECTION 4: Simple Queries
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Executing Simple Queries")
print("=" * 80)

# 4.1: Simple SELECT
console.print("\n[bold yellow]4.1: Simple SELECT query[/]")
query = "SELECT 1 as number, 'Hello Northwind' as message"
result = backend.execute_query(query)
console.print(result)

# 4.2: Query customers table
console.print("\n[bold yellow]4.2: Query customers table[/]")
query = """
SELECT customer_id, company_name, country
FROM customers
LIMIT 5
"""
result = backend.execute_query(query)
console.print(result)

# 4.3: Count records
console.print("\n[bold yellow]4.3: Count records in each table[/]")
counts_query = """
SELECT 'customers' as table_name, COUNT(*) as row_count FROM customers
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'employees', COUNT(*) FROM employees
"""
result = backend.execute_query(counts_query)
console.print(result)

# ============================================================================
# SECTION 5: Advanced Queries - JOINs
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Advanced Queries - JOINs")
print("=" * 80)

# 5.1: Two-table JOIN
console.print("\n[bold yellow]5.1: Orders with customer information[/]")
query = """
SELECT 
    o.order_id,
    o.order_date,
    c.company_name,
    c.country
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
ORDER BY o.order_date DESC
LIMIT 10
"""
result = backend.execute_query(query)
console.print(result)

# 5.2: Multi-table JOIN
console.print("\n[bold yellow]5.2: Order details with product and customer info[/]")
query = """
SELECT 
    o.order_id,
    c.company_name as customer,
    p.product_name,
    od.quantity,
    od.unit_price,
    (od.quantity * od.unit_price) as line_total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_details od ON o.order_id = od.order_id
JOIN products p ON od.product_id = p.product_id
LIMIT 10
"""
result = backend.execute_query(query)
console.print(result)

# ============================================================================
# SECTION 6: Aggregate Queries
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Aggregate Queries")
print("=" * 80)

# 6.1: Simple aggregates
console.print("\n[bold yellow]6.1: Order statistics[/]")
query = """
SELECT 
    COUNT(*) as total_orders,
    SUM(freight) as total_freight,
    AVG(freight) as avg_freight,
    MIN(freight) as min_freight,
    MAX(freight) as max_freight
FROM orders
"""
result = backend.execute_query(query)
console.print(result)

# 6.2: GROUP BY - Customers by country
console.print("\n[bold yellow]6.2: Customers grouped by country[/]")
query = """
SELECT 
    country,
    COUNT(*) as customer_count
FROM customers
GROUP BY country
ORDER BY customer_count DESC
LIMIT 10
"""
result = backend.execute_query(query)

# Display as Rich table
table = Table(title="Top 10 Countries by Customer Count")
table.add_column("Country", style="cyan")
table.add_column("Customer Count", justify="right", style="green")

for _, row in result.iterrows():
    table.add_row(row["country"], str(row["customer_count"]))

console.print(table)

# 6.3: Revenue by product
console.print("\n[bold yellow]6.3: Top 10 products by revenue[/]")
query = """
SELECT 
    p.product_name,
    COUNT(DISTINCT od.order_id) as times_ordered,
    SUM(od.quantity) as total_quantity,
    SUM(od.quantity * od.unit_price) as total_revenue
FROM products p
LEFT JOIN order_details od ON p.product_id = od.product_id
GROUP BY p.product_id, p.product_name
ORDER BY total_revenue DESC NULLS LAST
LIMIT 10
"""
result = backend.execute_query(query)
console.print(result)

# ============================================================================
# SECTION 7: Complex Queries - Subqueries
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: Complex Queries with Subqueries")
print("=" * 80)

console.print("\n[bold yellow]7.1: Products with order count (subquery)[/]")
query = """
SELECT 
    p.product_name,
    p.unit_price,
    (SELECT COUNT(*) 
     FROM order_details od 
     WHERE od.product_id = p.product_id) as order_count
FROM products p
ORDER BY order_count DESC
LIMIT 10
"""
result = backend.execute_query(query)
console.print(result)

# ============================================================================
# SECTION 8: Schema Introspection
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: Schema Introspection")
print("=" * 80)

# Create SchemaIntrospector
introspector = SchemaIntrospector(backend)

# Extract complete schema
console.print("\n[bold yellow]8.1: Extracting database schema[/]")
# NOTE: For PostgreSQL, use "public" as the schema name
schema = introspector.extract_database_schema("public")

console.print(f"✓ Database: {schema['database_name']}")
console.print(f"✓ Source: {schema['source']}")
console.print(f"✓ Table count: {schema['table_count']}")
console.print(f"✓ Extracted at: {schema['extracted_at']}")

# Show table details
console.print("\n[bold yellow]8.2: Table schemas[/]")
for table_name, table_info in sorted(schema["tables"].items())[:5]:
    console.print(
        f"\n[bold cyan]{table_name}[/] ({table_info['column_count']} columns)"
    )
    for col in table_info["columns"][:3]:  # Show first 3 columns
        console.print(f"  • {col['name']}: {col['type']}")

# Get schema statistics
console.print("\n[bold yellow]8.3: Schema statistics[/]")
stats = introspector.get_schema_statistics("public")
console.print(f"Total tables: {stats['table_count']}")
console.print(f"Total columns: {stats['total_columns']}")
console.print(f"Column types: {list(stats['column_types'].keys())[:5]}...")

# ============================================================================
# SECTION 9: Working with DataFrames
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: Working with Query Results as DataFrames")
print("=" * 80)

console.print("\n[bold yellow]9.1: Query results are pandas DataFrames[/]")
query = "SELECT * FROM customers LIMIT 10"
df = backend.execute_query(query)

console.print(f"Type: {type(df)}")
console.print(f"Shape: {df.shape}")
console.print(f"Columns: {list(df.columns)}")

# DataFrame operations
console.print("\n[bold yellow]9.2: DataFrame operations[/]")
console.print(f"First 3 rows:\n{df.head(3)}")
console.print(f"\nColumn types:\n{df.dtypes}")
console.print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# SECTION 10: Practical Examples
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: Practical Business Queries")
print("=" * 80)

# 10.1: Monthly sales trend
console.print("\n[bold yellow]10.1: Monthly sales trend[/]")
query = """
SELECT 
    DATE_TRUNC('month', o.order_date) as month,
    COUNT(DISTINCT o.order_id) as order_count,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    SUM(od.quantity * od.unit_price) as revenue
FROM orders o
JOIN order_details od ON o.order_id = od.order_id
WHERE o.order_date IS NOT NULL
GROUP BY month
ORDER BY month DESC
LIMIT 12
"""
result = backend.execute_query(query)
console.print(result)

# 10.2: Employee performance
console.print("\n[bold yellow]10.2: Employee sales performance[/]")
query = """
SELECT 
    e.first_name || ' ' || e.last_name as employee_name,
    e.title,
    COUNT(DISTINCT o.order_id) as orders_handled,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    SUM(od.quantity * od.unit_price) as total_sales
FROM employees e
JOIN orders o ON e.employee_id = o.employee_id
JOIN order_details od ON o.order_id = od.order_id
GROUP BY e.employee_id, employee_name, e.title
ORDER BY total_sales DESC
"""
result = backend.execute_query(query)
console.print(result)

# 10.3: Customer lifetime value
console.print("\n[bold yellow]10.3: Top customers by lifetime value[/]")
query = """
SELECT 
    c.company_name,
    c.country,
    COUNT(DISTINCT o.order_id) as order_count,
    SUM(od.quantity * od.unit_price) as lifetime_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_details od ON o.order_id = od.order_id
GROUP BY c.customer_id, c.company_name, c.country
ORDER BY lifetime_value DESC
LIMIT 15
"""
result = backend.execute_query(query)
console.print(result)

# ============================================================================
# SECTION 11: Context Manager Usage
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: Using AnalyticsConnector as Context Manager")
print("=" * 80)

console.print("\n[bold yellow]11.1: Context manager ensures proper cleanup[/]")

# Using with statement
with AnalyticsConnector(connection_string) as ctx_backend:
    result = ctx_backend.execute_query("SELECT COUNT(*) as total FROM customers")
    customer_count = result["total"][0]
    console.print(f"✓ Customer count: {customer_count}")

console.print("✓ Backend automatically closed after context")

# ============================================================================
# SECTION 12: Cleanup
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 12: Cleanup")
print("=" * 80)

# Close the backend connection
backend.close()
console.print("✓ [green]Backend connection closed[/]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("WALKTHROUGH COMPLETE!")
print("=" * 80)

summary = """
[bold green]✓ You've learned how to:[/]

1. Connect to PostgreSQL using AnalyticsConnector
2. Discover databases, schemas, and tables
3. Execute simple SELECT queries
4. Perform JOINs across multiple tables
5. Use aggregate functions (COUNT, SUM, AVG)
6. Write complex queries with subqueries
7. Introspect database schemas
8. Work with results as pandas DataFrames
9. Execute practical business queries
10. Use context managers for resource cleanup

[bold cyan]Next Steps:[/]
- Explore more complex queries
- Integrate with your own applications
- Try with different PostgreSQL databases
- Check out SchemaVectorizer for RAG applications

[bold yellow]Documentation:[/]
- AnalyticsConnector: src/connectors/analytics_backend.py
- SchemaIntrospector: src/utils/schema_introspector.py
- Test examples: tests/integration/test_northwind_*.py
"""

console.print(summary)

# ============================================================================
# APPENDIX: Quick Reference
# ============================================================================
print("\n" + "=" * 80)
print("APPENDIX: Quick Reference")
print("=" * 80)

reference = """
[bold]Connection String Format:[/]
postgres://user:password@host:port/database

[bold]Common Operations:[/]
• backend = AnalyticsConnector(connection_string)
• databases = backend.list_databases()
• tables = backend.list_tables(database=None)
• schema = backend.get_table_schema(table_name)
• result = backend.execute_query(sql_query)
• backend.close()

[bold]Context Manager:[/]
with AnalyticsConnector(connection_string) as backend:
    result = backend.execute_query("SELECT * FROM table")

[bold]Schema Introspection:[/]
introspector = SchemaIntrospector(backend)
schema = introspector.extract_database_schema("schema_name")
stats = introspector.get_schema_statistics("schema_name")

[bold]Northwind Database:[/]
• Host: localhost:5555
• Database: northwind
• Tables: 14 (customers, orders, products, etc.)
• Total Records: ~3,300
"""

console.print(reference)
