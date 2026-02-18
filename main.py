import numpy as np
dataSetSize = 1000
class Bias(object):
    def __init__(self) :
        self.w_Income = 0.5
        self.w_Age = 0.5
        self.w_Credits = 0.5
        self.Bias = 0.5


class User(object):
    sigma_income = 20000
    d_income = 50000

    sigma_age = 12
    d_age = 40

    sigma_credits = 2
    d_credits = 2
    def __init__(self, income, age, b_credits, credit_default):
        self.income = int(income)
        self.age = int(age)
        self.b_credits= int(b_credits)
        self.credit_default = int(credit_default)

    def verify(self):
        if int(self.income) <= 0:
            print("User has insufficient income")
        if int(self.age) <= 18:
            print(" is too young")
        if int(self.b_credits) < 0:
            print("Number of credits cant be less than 0")
            return False

    def get_inputs(self):
        x1 = (self.income - self.d_income) / self.sigma_income
        x2 = (self.age - self.d_age) / self.sigma_age
        x3 = (self.b_credits - self.d_credits) / self.sigma_credits
        return np.array([x1, x2, x3])

def sigmoid(z):
    return 1/ (1+ np.exp( -z ))
def sigmoid_deriv(z):
    return z*(1-z)





userlist = [
    User(11000,19,1, 1),
    User(20.54,43,0, 1),
    User(200000,35,3, 0)
]
for i in range (dataSetSize):
    userlist.append(User(np.random.randint(10000,80000000),np.random.randint(18,100), np.random.randint(50), np.random.choice([0,1])))
newBias = Bias()
for user in userlist:
    user.verify()

learning_rate = 0.1

for epoch in range(100):
    for user in userlist:
        # 1. Inputs holen
        x = user.get_inputs()  # [x1, x2, x3]
        target = user.credit_default

        # 2. Forward Pass (Vorhersage)
        # z = w1*x1 + w2*x2 + w3*x3 + b
        z = (x[0] * newBias.w_Income +
             x[1] * newBias.w_Age +
             x[2] * newBias.w_Credits +
             newBias.Bias)
        prediction = sigmoid(z)

        # 3. Backpropagation (Fehler berechnen)
        error = target - prediction

        # Der Gradient (Fehler * Steigung der Sigmoid-Kurve)
        d_z = error * sigmoid_deriv(prediction)

        # 4. Update der Gewichte (Schuldzuweisung)
        # Jedes Gewicht wird anteilig an seinem Input korrigiert
        newBias.w_Income += learning_rate * d_z * x[0]
        newBias.w_Age += learning_rate * d_z * x[1]
        newBias.w_Credits += learning_rate * d_z * x[2]
        newBias.Bias += learning_rate * d_z

# Test-Ausgabe
print(f"Neue Gewichte nach Training: Income: {newBias.w_Income:.2f}, Age: {newBias.w_Age:.2f}")