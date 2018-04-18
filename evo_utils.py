from properties import Properties


def getRandomProps(n = 100):
    p = Properties(10, 10,  75, 0.2)
    for i in range(n):
        p.mutate()
    return p