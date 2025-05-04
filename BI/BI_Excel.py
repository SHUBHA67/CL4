# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# Step 1: Load Excel Data
# ----------------------------------------

# Read all sheets from Excel
excel_file = "sales_data.xlsx"
sales_df = pd.read_excel(excel_file, sheet_name="Sales")
customers_df = pd.read_excel(excel_file, sheet_name="Customers")
products_df = pd.read_excel(excel_file, sheet_name="Products")

print("Sales Data:")
display(sales_df.head())

print("\nCustomers Data:")
display(customers_df.head())

print("\nProducts Data:")
display(products_df.head())

# ----------------------------------------
# Step 2: Data Cleaning (Excel-like "Format as Table")
# ----------------------------------------

# Remove duplicates
sales_df.drop_duplicates(inplace=True)

# Handle missing values (e.g., fill with 0)
sales_df["Quantity"].fillna(0, inplace=True)

# Convert OrderDate to datetime
sales_df["OrderDate"] = pd.to_datetime(sales_df["OrderDate"])

# ----------------------------------------
# Step 3: Advanced Excel Formulas (VLOOKUP, SUMIF)
# ----------------------------------------

# VLOOKUP: Merge Customer Name into Sales Data
sales_df = pd.merge(sales_df, customers_df, on="CustomerID", how="left")

# VLOOKUP: Merge Product Price into Sales Data
sales_df = pd.merge(sales_df, products_df[["Product", "Price"]], on="Product", how="left")

# SUMIF: Calculate Total Revenue (Quantity * Price)
sales_df["TotalRevenue"] = sales_df["Quantity"] * sales_df["Price"]

print("\nMerged Data with Revenue:")
display(sales_df.head())

# ----------------------------------------
# Step 4: PivotTables (Excel-like Summarization)
# ----------------------------------------

# Create a PivotTable: Total Revenue by Region and Product
pivot_table = pd.pivot_table(
    sales_df,
    values="TotalRevenue",
    index="Region",
    columns="Product",
    aggfunc="sum",
    fill_value=0
)

print("\nPivotTable (Revenue by Region & Product):")
display(pivot_table)

# ----------------------------------------
# Step 5: Visualization (Excel-like Charts)
# ----------------------------------------

# Bar Chart: Total Revenue by Product
plt.figure(figsize=(10, 6))
sns.barplot(data=sales_df, x="Product", y="TotalRevenue", estimator=sum, ci=None)
plt.title("Total Revenue by Product (Excel-like Bar Chart)")
plt.xlabel("Product")
plt.ylabel("Total Revenue ($)")
plt.show()

# Line Chart: Monthly Sales Trend
sales_df["Month"] = sales_df["OrderDate"].dt.month_name()
monthly_sales = sales_df.groupby("Month")["TotalRevenue"].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_sales, x="Month", y="TotalRevenue", marker="o")
plt.title("Monthly Sales Trend (Excel-like Line Chart)")
plt.xlabel("Month")
plt.ylabel("Total Revenue ($)")
plt.show()

# ----------------------------------------
# Step 6: Export Results to Excel
# ----------------------------------------

with pd.ExcelWriter("analysis_results.xlsx") as writer:
    sales_df.to_excel(writer, sheet_name="Processed Sales", index=False)
    pivot_table.to_excel(writer, sheet_name="PivotTable")