class House:
    def __init__(self, id,date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,
                 grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15):
        self.id = id
        self.date = date
        self.price = price
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.sqft_living = sqft_living
        self.sqft_lot = sqft_lot
        self.floors = floors
        self.waterfront = waterfront
        self.view = view
        self.condition = condition
        self.grade = grade
        self.sqft_above = sqft_above
        self.sqft_basement = sqft_basement
        self.yr_built = yr_built
        self.yr_renovated = yr_renovated
        self.zipcode = zipcode
        self.lat = lat
        self.long = long
        self.sqft_living15 = sqft_living15
        self.sqft_lot15 = sqft_lot15
        self.yearBin = 0

    def isValid(self):
        for attr, value in self.__dict__.items():
            if value == "" or value is None:
                return False
        return True

    def returnAtribute(self, atribute):
        for attr, value in self.__dict__.items():
            if attr == atribute:
                return value
    def checkAtribute(self, atribute, total):
        for attr, value in self.__dict__.items():
            if attr == atribute:
                if value:
                    return total
                else:
                    return total+1