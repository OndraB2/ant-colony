from re import L
import numpy as np
import random
from math import ceil
import matplotlib.pyplot as plt
import string

class TSP:
    def __init__(self) -> None:
        self.NP = 40
        self.G = 300
        self.city_count = 15  # In TSP, it will be a number of cities
        self.min = 0
        self.max = 200

        self.cities = []

    def distance(self, a, b):
        return np.linalg.norm(b-a)

    def swap(self, index1, index2, l: list):
        temp = l[index1]
        l[index1] = l[index2]
        l[index2] = temp

        return l


    def mutate(self, l):
        index1 = random.randint(1, len(l) - 2)
        index2 = random.randint(1, len(l) - 2)

        while index1 == index2:
            index2 = random.randint(1, len(l) - 2)

        return self.swap(index1, index2, l)
            
    def crossover(self, a: list, b: list):
        length = len(a)
        # first_half_count = ceil(length / 2)
        first_half_count = random.randint(1, len(a)-1)
        cross = a[:first_half_count]

        for x in b:
            if x not in cross:
                cross.append(x)

        cross.append(a[0])

        return cross


    def generateCities(self):
        l = []
        first = random.uniform(self.min, self.max)

        for i in range(self.city_count):
            city_x = random.uniform(self.min, self.max)  
            city_y = random.uniform(self.min, self.max)  

            while (city_x, city_y) in l:    # generuj dokud mesto nebude unikatni
                city_x = random.uniform(self.min, self.max)  
                city_y = random.uniform(self.min, self.max)  

            l.append((city_x, city_y))    

        return l   

    def generatePopulation(self):
        self.cities = self.generateCities()

        l = []
        for i in range(self.NP):
            cities_indexes = random.sample(range(self.city_count), self.city_count) # vyygeneruju jednu populaci mest
            temp_l = []
            for index in cities_indexes:
                temp_l.append(self.cities[index])

            first = temp_l[0]
            temp_l.append(first)  # aby list zacinal a koncil stejnym mestem  

            l.append(temp_l)
        return l

    def evaluateCities(self, cities):
        sum = 0
        for j in range(self.city_count): # ikdyz jdu po dvojickach, ta nemusim davat -1, protoze mam prvni mesto i na konci
            sum += self.distance( np.array(cities[j]), np.array(cities[j+1]) ) 
        return sum

    def evaluatePopulation(self, population):
        l = []
        for cities in population:
            sum = self.evaluateCities(cities)
            l.append(sum)
        return l 

    def splitPoints(self, cities: list):
        x = []
        y = []
        for city in cities:
            x.append(city[0])
            y.append(city[1])
        return x, y

    def visibilityMatrix(self, cities, col_to_null):
        matrix = np.zeros((self.city_count, self.city_count))
        for i in range(self.city_count):
            for j in range(self.city_count):
                if i != j and j != col_to_null:
                    matrix[i][j] = 1 / self.distance(np.array(cities[i]), np.array(cities[j]))

        return matrix

    def algorithm(self):
        x = range(tsp.min, tsp.max + 50, 50)
        y = range(tsp.min, tsp.max + 50, 50)

        vapo_coef = 0.5
        alpha = 1
        beta = 2

        scat = plt.scatter(x, y)
        scat.remove()
        plt.ion()
        ax = plt.gca()
        scat = ax.scatter(x, y)
        scat.remove()

        cities = self.generateCities()

        phero_mat = np.ones((self.city_count, self.city_count))

        x, y = self.splitPoints(cities)

        scat, = ax.plot(x, y, '-o')
        
        for i in range(len(x)-1):
            ax.annotate(i, (x[i]+1, y[i]+1))
        
        plt.pause(0.1)
        scat.remove()

        # give each ant different starting city
        ants_path = [[city] for city in cities]
        ants_path_indexes = [[i] for i, _ in enumerate(cities)]


        for i in range(self.G):
            ants_path = [[city] for city in cities]
            ants_path_indexes = [[i] for i, _ in enumerate(cities)]
            for ant_i in range(self.city_count): # for each ant
                vis_mat = self.visibilityMatrix(cities, ant_i)
                row_i = ant_i
                for city_i in range(self.city_count-1): # for each row in visivility mat, -1 because last iteration we only have one city
                    l = []
                    for col_i in range(self.city_count): # for each col in row_i row
                        l.append(phero_mat[row_i][col_i]**alpha * vis_mat[row_i][col_i]**beta)
                    sum_of_l = sum(l)
                    
                    # calculate probability
                    prob = [x / sum_of_l for x in l]

                    # calculate cumulative probability
                    cumul = [sum(prob[:index]) for index in range(1, self.city_count+1)]

                    r = np.random.uniform()

                    # choose city to travel
                    city_i_to_travel = 0
                    while r > cumul[city_i_to_travel]:
                        city_i_to_travel += 1
                    ants_path[ant_i].append(cities[city_i_to_travel])
                    ants_path_indexes[ant_i].append(city_i_to_travel)

                    row_i = city_i_to_travel

                    # null row in vis matrix
                    for row_index in range(self.city_count):
                        vis_mat[row_index][city_i_to_travel] = 0

            # append starting city to end and evaluate path
            paths_eval = []
            for ant_i in range(self.city_count):
                ants_path[ant_i].append(ants_path[ant_i][0]) 
                ants_path_indexes[ant_i].append(ants_path_indexes[ant_i][0])
                paths_eval.append(self.evaluateCities(ants_path[ant_i]))

            # update pheromone matrix
            # multiply all cells by vaporization coeficient
            for i_index in range(self.city_count):
                for j_index in range(self.city_count):
                    phero_mat[i_index][j_index] *= vapo_coef

            # add path eval to cities ant went through
            for ant_i in range(self.city_count):
                for k in range(self.city_count): 
                    row_index = ants_path_indexes[ant_i][k]    
                    col_index = ants_path_indexes[ant_i][k+1]    

                    phero_mat[row_index][col_index] += 1 / paths_eval[ant_i]

            # get best path
            min_val = paths_eval[0]
            min_index = 0
            for index, x in enumerate(paths_eval):
                if x < min_val:
                    min_val = x
                    min_index = index

            print(paths_eval[min_index])
            print('\n')

            x, y = self.splitPoints(ants_path[min_index])

            if i < self.G -1:
                scat, = ax.plot(x, y, '-o')
                plt.title('Gen: ' + str(i))
                plt.pause(0.02)
                scat.remove()
            else:
                scat, = ax.plot(x, y, '-o')
                plt.title('Gen: ' + str(i))
                plt.pause(20)        
 
tsp = TSP()
tsp.algorithm()







