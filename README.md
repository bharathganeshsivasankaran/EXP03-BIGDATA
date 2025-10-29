# EXP03-BIGDATA
```
Name : Bharathganesh S
Reg No : 212222230022
```
# 1 - Number Analysis with Dataset

## AIM:
To analyze a list of numbers using PySpark to identify even, odd, prime, and palindromic numbers, and compute basic statistics.

## PROCEDURE:
```
Initialize SparkSession.

Load the numbers.csv file.

Filter even and odd numbers using modulo %.

Compute statistics — max, min, sum, average.

Define a vectorized UDF to identify prime numbers.

Define another pandas UDF to detect palindromic numbers.

Use RDD operations to calculate total sum and even count.

Display all results using .show().
```

## PROGRAM:
```py
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("Number_Analysis_Extended").getOrCreate()

# Read CSV
nums = spark.read.option("header", True).option("inferSchema", True).csv("data/numbers.csv")

# Even and Odd split
even = nums.filter((F.col("Value") % 2) == 0)
odd = nums.filter((F.col("Value") % 2) != 0)

print("Evens:"); even.show()
print("Odds:"); odd.show()

# Statistics
stats = nums.agg(
    F.max("Value").alias("max_val"),
    F.min("Value").alias("min_val"),
    F.sum("Value").alias("sum_val"),
    F.round(F.avg("Value"), 2).alias("avg_val")
)
print("Stats:")
stats.show()

# -------- Vectorized Prime Check using pandas_udf --------
@pandas_udf(BooleanType())
def is_prime_pandas(series: pd.Series) -> pd.Series:
    def check_prime(n):
        if n is None or n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True
    return series.apply(check_prime)

# Apply vectorized UDF
primes = nums.filter(is_prime_pandas(F.col("Value")))
print("Primes:")
primes.show()

# -------- Palindromic Number Detection --------
@pandas_udf(BooleanType())
def is_palindrome(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda x: x == x[::-1])

palindromes = nums.filter(is_palindrome(F.col("Value")))
print("Palindromes:")
palindromes.show()

# -------- RDD Operations (for comparison) --------
rdd = nums.rdd.map(lambda row: row["Value"])
total_sum = rdd.reduce(lambda a, b: a + b)
even_count = rdd.filter(lambda x: x % 2 == 0).count()

print("RDD Summary:")
print("Sum =", total_sum, "Even Count =", even_count)

```
## OUTPUT:

<img width="1002" height="406" alt="image" src="https://github.com/user-attachments/assets/fce53417-4440-4a0d-b88e-43244a28556a" />

<img width="1041" height="367" alt="image" src="https://github.com/user-attachments/assets/8177128c-ceec-4e99-ac77-b74b6639bdd4" />


## RESULT:
Thus, the program to analyze a list of numbers using PySpark to identify even, odd, prime, and palindromic numbers, and compute basic statistics has been successfully completed


# 2 - Logical Analysis on Age Data

## AIM:
To categorize people as Minor, Adult, or Senior based on age and find age statistics.

## PROCEDURE:
```
Initialize SparkSession.

Load the people.csv dataset.

Create a new column "Category" using when().otherwise().

Count people in each category with groupBy().count().

Find the oldest and youngest using orderBy().

Compute median age using approxQuantile().

Convert data to Pandas using .toPandas().

Plot age distribution using matplotlib.
```

## PROGRAM:
```py
from pyspark.sql import SparkSession, functions as F
import matplotlib.pyplot as plt

# Initialize Spark
spark = SparkSession.builder.appName("People_Categorization_Extended").getOrCreate()

# Read data
people = spark.read.option("header", True).option("inferSchema", True).csv("persons.csv")

# 1️⃣ Categorize people
people_cat = people.withColumn(
    "Category",
    F.when(F.col("Age") < 18, "Minor")
     .when((F.col("Age") >= 18) & (F.col("Age") <= 59), "Adult")
     .otherwise("Senior")
)
print("People with Category:")
people_cat.show()

# 2️⃣ Count per category
print("Count per Category:")
counts = people_cat.groupBy("Category").count()
counts.show()

# 3️⃣ Oldest and Youngest
print("Oldest:")
oldest = people.orderBy(F.desc("Age")).limit(1)
oldest.show()

print("Youngest:")
youngest = people.orderBy(F.asc("Age")).limit(1)
youngest.show()

# 4️⃣ Median Age (approximate)
median_age = people.approxQuantile("Age", [0.5], 0.01)[0]
print(f"Median Age ≈ {median_age}")

# 5️⃣ Export for plotting (optional visualization)
pdf = people.toPandas()

# Plot age distribution
plt.figure(figsize=(6,4))
plt.hist(pdf["Age"], bins=5, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

```

## OUTPUT:

<img width="974" height="330" alt="image" src="https://github.com/user-attachments/assets/921d38f2-86f5-46dc-ada7-47ba04bd844d" />

<img width="786" height="234" alt="image" src="https://github.com/user-attachments/assets/3afa6e50-de2c-4746-b5b0-267508a27978" />

<img width="992" height="397" alt="image" src="https://github.com/user-attachments/assets/0293775b-251c-42bc-993e-8e71e1061e48" />

## RESULT:
Thus the program to categorize people as Minor, Adult, or Senior based on age and find age statistics.



# 3 - Product Sales Analysis

## AIM:
To analyze product sales and identify top-selling products, categories, and reorder requirements.

## PROCEDURE:
```
Initialize SparkSession.

Load sales.csv.

Add a Revenue column as Quantity * Price.

Group by Product and Category to find totals.

Identify best-selling product and category.

Filter products with low total quantity (<100).

Compute monthly revenue if Date exists.

Mark Reorder_Required for items below threshold (e.g., 50).
```
## PROGRAM:
```py
from pyspark.sql import SparkSession, functions as F

# Initialize Spark
spark = SparkSession.builder.appName("Sales_Analysis_Extended").getOrCreate()

# Read CSV
sales = spark.read.option("header", True).option("inferSchema", True).csv("products.csv")

# 1️⃣ Compute Revenue per product
sales_with_rev = sales.withColumn("Revenue", F.col("Quantity") * F.col("Price"))
rev_per_product = sales_with_rev.groupBy("Product", "Category") \
    .agg(
        F.sum("Revenue").alias("total_revenue"),
        F.sum("Quantity").alias("total_qty")
    )

print("Revenue per Product:")
rev_per_product.orderBy(F.desc("total_revenue")).show()

# 2️⃣ Best-selling product and category
print("Best-Selling Product:")
best_product = rev_per_product.orderBy(F.desc("total_qty")).limit(1)
best_product.show()

print("Best-Selling Category:")
best_category = sales.groupBy("Category") \
    .agg(F.sum("Quantity").alias("category_qty")) \
    .orderBy(F.desc("category_qty")).limit(1)
best_category.show()

# 3️⃣ Products with sales below 100 units
print("Low-Sales Products (Total Qty < 100):")
low_sales_products = rev_per_product.filter(F.col("total_qty") < 100)
low_sales_products.show()

# -------------------------------------------------------------
# 4️⃣ Extension 1: Monthly revenue (if Date column exists)
# -------------------------------------------------------------
if "Date" in sales.columns:
    monthly_revenue = sales_with_rev.withColumn("Month", F.date_format(F.col("Date"), "yyyy-MM")) \
        .groupBy("Month") \
        .agg(F.sum("Revenue").alias("monthly_revenue")) \
        .orderBy("Month")
    print("Monthly Revenue:")
    monthly_revenue.show()
else:
    print("No 'Date' column found — skipping monthly revenue calculation.")

# -------------------------------------------------------------
# 5️⃣ Extension 2: Reorder suggestion (based on threshold)
# -------------------------------------------------------------
threshold = 50
reorder_suggestions = rev_per_product.withColumn(
    "Reorder_Required",
    F.when(F.col("total_qty") < threshold, "YES").otherwise("NO")
)

print("Reorder Suggestions:")
reorder_suggestions.orderBy("Product").show()

# -------------------------------------------------------------
# 6️⃣ (Optional) Save results to CSV
# -------------------------------------------------------------
reorder_suggestions.coalesce(1).write.mode("overwrite").option("header", True).csv("output/reorder_summary")

print("✅ Results saved to output/reorder_summary folder.")

```
## OUTPUT:
<img width="1162" height="400" alt="image" src="https://github.com/user-attachments/assets/3e05299c-5415-4266-b80d-9f4ae0ed7cd5" />

<img width="1032" height="387" alt="image" src="https://github.com/user-attachments/assets/a387b035-ab10-48b6-9830-620ac3b77fe0" />


## RESULT:
Thus the program to analyze product sales and identify top-selling products, categories, and reorder requirements.

