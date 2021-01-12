class TestFunction:
    def __init__(self, function, name):
        self.function = function
        self.name = name
    
    def __call__(self, x):
        return self.function(x)

    def __str__(self):
        return self.name

test_collection = []
test_collection.append(TestFunction(
    lambda x : x,
    "x")
    )

test_collection.append(TestFunction(
    lambda x : x **2,
    "x^2")
    )

