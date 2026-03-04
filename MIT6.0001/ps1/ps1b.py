annual_salary = float(input("Enter annual salary:"))
portion_saved = float(input("Enter portion saved:"))
total_cost = float(input("Enter total cost:"))
semi_annual_raise = float(input("Enter semi_annual_raise:"))
portion_down_payment = 0.25

current_savings = 0
month = 0
r = 0.04
#month_count = 0
while current_savings < portion_down_payment * total_cost:
    month += 1
    current_savings += current_savings * r / 12 + (annual_salary / 12) * portion_saved
    
    if month % 6 == 0:
        annual_salary += annual_salary * semi_annual_raise

print(month)