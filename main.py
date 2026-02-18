import numpy as np
dataSetSize = 1000
class Bias(object):
    def __init__(self) :
        self.w_Income = np.random.randn()
        self.w_Age = np.random.randn()
        self.w_Credits = np.random.randn()



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

    def defaultIncome(self):
        tempIncome = (self.income - self.d_income) / self.sigma_income
        return tempIncome
    def defaultAge(self):
        tempAge = (self.age - self.d_age) / self.sigma_age
        return tempAge
    def defaultCredits(self):
        tempCredits = (self.b_credits - self.d_credits) / self.sigma_credits
        return tempCredits

    def logisticLog(self, expectedBias : Bias):
        z = self.defaultIncome() * expectedBias.w_Income + self.defaultAge() * expectedBias.w_Age + self.defaultCredits() * expectedBias.w_Credits
        return 1/ (1+ np.exp( -z ))

    def lossFunction(self, expectedBias : Bias):
        if self.credit_default == 1 :
            loss = np.log(self.logisticLog(expectedBias))
            return loss
        if  self.credit_default == 0 :
            loss = (np.log(self.logisticLog(expectedBias)) ) * -1
            return loss




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
for user in userlist:
    print("Chance eines defaults:" , user.logisticLog(newBias))