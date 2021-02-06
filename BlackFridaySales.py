from pydantic import BaseModel

class BlackFridaySale(BaseModel):
    Gender: int
    Age: int
    Occupation: int
    City_Category: int
    Stay_In_Current_City_Years: int
    Marital_Status:int
    Product_Category_1 : int
    Product_Category_2 : int
    Product_Category_3 : int
