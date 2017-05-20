# -*- coding: utf-8 -*-

"""
The input grid can be included directly in the script.
It will be taken care of shortly
Written the 20th May 2017
"""

############## Libraries to import ########

import random
from operator import add
import numpy as np
from itertools import chain
import time

############## Parameters #################

COUNT=20
LENGTH_EXCEEDED=150
POPULATION = 20
GENERATIONS = 1000000
DEBUG_STEP = 1000
random.seed(0)

###########################################

inputt=[[0 ,0 ,0 ,0 ,0 ,0 ,8 ,1 ,0],
        [0 ,4 ,0 ,8 ,0 ,6 ,0 ,0 ,0],
        [0 ,0 ,5 ,0 ,0 ,2 ,0 ,9 ,0],
        [8 ,7 ,0 ,4 ,0 ,1 ,9 ,0 ,0],
        [0 ,0 ,0 ,0 ,7 ,0 ,0 ,0 ,0],
        [0 ,0 ,4 ,5 ,0 ,9 ,0 ,8 ,3],
        [0 ,6 ,0 ,1 ,0 ,0 ,4 ,0 ,0],
        [0 ,0 ,0 ,6 ,0 ,5 ,0 ,2 ,0],
        [0 ,2 ,8 ,0 ,0 ,0 ,0 ,0 ,0]]


#### Snippet to fix
#import os 
#import re

#def purify(el):
#  return re.sub('\D','',el)

#def extract_inputs_from_files():

#  complete_inputs=[]
#  files = [f for f in os.listdir('.') if os.path.isfile(f)]
#  for f in files :  
#    array=[]
#    with open(f, "r") as output:
#           for line in output:     
#              line= map(purify,line.split(' '))
#              print line
#        if not re.findall('\d',line[0]):
#          break
#        line=[int(el) for el in line]
#              if array :
#          array.append(line)
#        print array
#          complete_inputs.append(array)
#  
#    return complete_inputs


def decompose_grid_thirds(grid):
  ### grid must be array
  list_grids=[]
  row,col=0,0
  for row in range(9):
      for col in range(9):
          if (col+1)%3==0 and (row+1)%3==0 and row*col>0:
              grid_part=grid[row-2:row+1,col-2:col+1].tolist()
              grid_to_put=[]
              for part in grid_part:
                    grid_to_put.extend(part)
              list_grids.append(grid_to_put)

  return list_grids

def decompose_grid_thirds_array(grid):
  return np.array(decompose_grid_thirds(grid))

### make sure the input respects certain requirements

def clean_input():
  global inputt
  first_grid=np.array(inputt)
  second_grid=[]
  for el in first_grid:
      el=[0 if el.tolist().count(element) > 1 else element for element in el]
      second_grid.append(el)

  final_grid=[]
  for el in np.transpose(np.array(second_grid)):
      el=[0 if el.tolist().count(element) > 1 else element for element in el]
      final_grid.append(el)

  return np.transpose(np.array(final_grid))

def draw_grid(grid):
    row,col=0,0
    for row in range(9):
          if row%3==0 and row>0:
                  print '---------------------'
          for col in range(9):
              if col%3==0 and col>0:
                  print '|',
                  print grid[row][col],
              else :
                  print grid[row][col],
          print '\n'     
    return ' The finally obtained grid ! \n' 

##### individual
def individual(minimum,maximum):
    indiv=[]
    for _ in range(9**2-np.count_nonzero(clean_input())):
        indiv.append(random.randint(1,9))
    return indiv

##### population
def population(count,minimum,maximum):
    return [individual(minimum,maximum) for x in xrange(count) ]

##### shape individuals into a testable ones
def merge(individual):
    row,col=0,0
    cobaye=clean_input()
    index_individual=0
    for row in range(9):
        for col in range(9):
            if not cobaye[row][col]:
                cobaye[row][col]=individual[index_individual]
                index_individual+=1
    return np.array(cobaye)


def fitness(individual):

    cobaye_array=merge(individual)
    test1=sum(9-len(set(el)) for el in cobaye_array)
    test2=sum(9-len(set(el)) for el in np.transpose(cobaye_array))
    test3=sum(9-len(set(el)) for el in decompose_grid_thirds_array(cobaye_array))
    return test1+test2+test3

def test_fitness(array):

      test1=sum(9-len(set(el)) for el in array)
      test2=sum(9-len(set(el)) for el in np.transpose(array))
      test3=sum(9-len(set(el)) for el in decompose_grid_thirds_array(array))
      return test1+test2+test3

##### a metric for a new generation
def grade(pop):

     summed = reduce(lambda x,y: x+y,(fitness(x) for x in pop))
     return summed / float(len(pop))

def first_elements(tulip):
    return tulip[1]

def evolve(pop, minimum,maximum, retain=.2, random_select=0.05, mutate_single=0.01):

    graded = [ (fitness(x), x) for x in pop]
    graded=map(first_elements,sorted(graded))
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    
    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)
          
    # mutate some individuals
    for individual in parents:
       
           if mutate_single > random.random():

              pos_to_mutate = random.randint(0, len(individual)-1)
            
              # The next two mutations are convenient
              # However, in order to achieve complete random mutations
              # We use the second procedure 

              #individual[pos_to_mutate] = random.randint(
              #   min(individual), max(individual))

              individual[pos_to_mutate] = random.randint(
                 minimum, maximum)

  
    # crossover parents to create children

    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    population_length=len(pop)
    children = []
    while len(children) < desired_length:

        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)
        if male != female:

            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]

            children.append(child)

    parents.extend(children)
    return parents

def individual_with_minimum_fitness_so_far(fitness_history,all_populations) :
  try :

    wanted_population = all_populations[-1]
    individual_with_minimum_fitness_index=wanted_population.index(min(wanted_population))
    wanted_individual = wanted_population[individual_with_minimum_fitness_index]

  except ValueError :
    pass
    return None

  return merge(wanted_individual)

if __name__=='__main__' : 

  pop=population(COUNT,1,9)
  fitness_history = []
  all_populations=[]
  check_and_wrangle=[]

  begin=time.time()
  for i in range(GENERATIONS):
    if i%DEBUG_STEP==0 and i>0 :
        debug=individual_with_minimum_fitness_so_far(fitness_history,all_populations)
        print 'Debug: \n',debug
    new_population = evolve(pop,1,9)
    check_and_wrangle.append(int(grade(new_population)))

    if len(check_and_wrangle) > LENGTH_EXCEEDED :
        check_and_wrangle.pop(0)

        if len(set(check_and_wrangle))==1:
            new_population=all_populations[-LENGTH_EXCEEDED]
            check_and_wrangle = []

    fitness_history.append(grade(new_population))
    all_populations.append(new_population)

    print '{0} score / progress : {1} %'.format(grade(new_population),(i*100/float(GENERATIONS)))
    pop=new_population
  end=time.time()

  print 'Sudoku to Solve : \n',clean_input()
  print 'Best individual obtained : \n',draw_grid(individual_with_minimum_fitness_so_far(fitness_history,all_populations))
  print 'Fitness of best individual \n: ',test_fitness(individual_with_minimum_fitness_so_far(fitness_history,all_populations))
  print 'time taken : ',end-begin

