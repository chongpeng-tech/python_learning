total_cost = 1000000
semi_annual_raise = 0.07
r = 0.04 #存款利率
months = 36

annual_salary  = float(input("Enter annual salary:"))

low = 0
high = 10000

def calculate_savings(annual_salary, portion_saved):
    current_savings = 0.0
    month = 0
    while month < 36:
        month += 1
        current_savings += current_savings * r / 12 + (annual_salary / 12) * portion_saved
        if month % 6 == 0:
            annual_salary += annual_salary * semi_annual_raise
    return current_savings

test = calculate_savings(150000, 0.1)
print(test)
            