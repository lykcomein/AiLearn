---
name: sql-formatter
description: 当用户需要格式化 SQL 语句、优化 SQL 可读性、或检查 SQL 语法规范时使用此技能。典型触发词：SQL、查询、格式化、美化、优化可读性。
---

# SQL 格式化技能

## 何时使用
- 用户贴出一段混乱的 SQL，希望排版美观
- 用户询问 SQL 写法是否符合规范
- 需要把单行 SQL 拆成多行便于阅读

## 格式化规则

### 1. 关键字大写
`SELECT`、`FROM`、`WHERE`、`JOIN`、`GROUP BY`、`ORDER BY`、`LIMIT` 等关键字统一大写。

### 2. 子句换行
每个主要子句独立一行：
```sql
SELECT a.id, a.name, COUNT(b.order_id) AS order_cnt
FROM users a
LEFT JOIN orders b ON a.id = b.user_id
WHERE a.status = 'active'
  AND a.created_at >= '2024-01-01'
GROUP BY a.id, a.name
HAVING COUNT(b.order_id) > 0
ORDER BY order_cnt DESC
LIMIT 100;
```

### 3. 多列缩进
当一行列过多时，每列独占一行，逗号置于行首或行尾（保持一致）。

### 4. JOIN 对齐
`ON` 条件与 JOIN 对齐，多个条件用 `AND` 换行：
```sql
LEFT JOIN orders b
  ON a.id = b.user_id
 AND b.status = 'paid'
```

## 输出规范
- 直接给格式化后的 SQL，用 ```sql 代码块
- 如果发现明显的性能问题（缺索引提示、SELECT *、笛卡尔积），在代码块后附一段"优化建议"
- 不要改变 SQL 的语义，仅调整排版