# PostgreSQL Indexing Basics

Indexes speed up reads by maintaining sorted or specialized structures separate from heap tables. Choosing the right index type and predicate matters for both latency and write amplification.

## B-tree indexes (default)

PostgreSQL's default **B-tree** index supports:

- Equality lookups (`WHERE status = 'active'`)
- Range scans (`WHERE created_at > '2024-01-01'`)
- Sorting on indexed columns

For single-column equality filters, a plain B-tree index is usually the first choice.

## Partial indexes

A **partial index** indexes only rows matching a `WHERE` clause:

```sql
CREATE INDEX idx_active_users ON users (email)
WHERE deleted_at IS NULL;
```

Use partial indexes when queries always filter on the same predicate and you want a smaller, faster index with less maintenance overhead.

## Composite indexes

Multi-column B-tree indexes help queries that filter or sort on leading columns. Put the most selective or most frequently filtered column first when designing composite keys.

## When to skip indexing

Avoid indexing low-cardinality columns alone, very wide rows, or tables with heavy write rates unless profiling shows a clear benefit.
