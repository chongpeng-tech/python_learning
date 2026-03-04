annual_salary = float(input("Enter annual salary:"))
portion_saved = float(input("Enter portion saved:"))
total_cost = float(input("Enter total cost:"))

portion_down_payment = 0.25

current_savings = 0
month = 0
r = 0.04

while current_savings < portion_down_payment * total_cost:
    current_savings += current_savings * r / 12 + (annual_salary / 12) * portion_saved
    month += 1

print(month)