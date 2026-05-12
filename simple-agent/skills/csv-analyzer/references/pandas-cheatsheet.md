# Pandas 常用操作速查

## 读取
```python
df = pd.read_csv("data.csv")
df.head(5)            # 前 5 行
df.info()             # 列名 + 类型 + 非空数
df.describe()         # 数值列统计
```

## 选择
```python
df["col"]                       # 单列
df[["a", "b"]]                  # 多列
df[df["age"] > 18]              # 行过滤
df.loc[10:20, ["a", "b"]]       # 区段
```

## 聚合
```python
df.groupby("city")["sales"].sum()
df.groupby("city").agg({"sales": ["sum", "mean"], "age": "max"})
```

## 缺失值
```python
df.isnull().sum()               # 每列空值数
df.dropna(subset=["age"])       # 删空
df.fillna({"age": df["age"].median()})  # 中位数填充
```