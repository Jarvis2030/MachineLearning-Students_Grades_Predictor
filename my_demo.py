from numpy import *
from numpy import average
from numpy import random
from numpy import genfromtxt  # to get the data from the data file
from numpy import gradient


def compute_error_for_line_given_points(b, m, points):
    # This is part of the gradient descent which minimizes the error of our line
    # initialize the error at 0

    totalError = 0
    # for every point
    for i in range(0, len(points)):
        # get the x values
        x = points[i, 0]
        y = points[i, 1]

        # get the difference, square it and add it to the total
        # this is the actual equation
        # formular link = "https://medium.com/meta-design-ideas/linear-regression-by-using-gradient-descent-algorithm-your-first-step-towards-machine-learning-a9b9c0ec41b1"

        totalError += (y - (m * x + b)) ** 2

    # get the average
    return totalError / float(len(points))


# Now declare a method to calculate the gradient descent using every single point
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting_b and m
    b = starting_b
    m = starting_m

    # gradient_descent
    for i in range(num_iterations):  # for every iteration, wea re going to perform a gradient descent
        # update b and m with the new more accurate b and m values by
        # performing this gradient steps
        b, m = step_gradient(b, m, array(points), learning_rate)

    # at the end of the gradient_descent_runner(), we are gonna return the optimal b and m
    return [b, m]  # to give us the line of best fit


# magic, the greatest the greatest
# given our current b and m values
def step_gradient(b_current, m_current, points, learningRate):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):  # We are iterating through and collecting any single point in our scatter plot
        # starting points for our gradient
        x = points[i, 0]
        y = points[i, 1]

        # now get the direction in respect to b and m

        # computing the partial derivatives of our error functions w.r.t b and w.r.t m
        # to give us the direction or both the b and m values for the minimum error in the direction search box

        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    # update our b and m values using our partial derivatives

    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]  # for the 1000 iterations we had


def run():  # This shows what we are doing at high level

    # Step 1- collect our data as the rule of ML
    points = genfromtxt('data.csv', delimiter=',')
    # points is taken a bunch of the x and y values.
    # x = study
    # y = our scores
    # we are creating two loops: 1: To convert each line of the code with sequence of strings and the
    # 2: converts each string into appropriate data type
    # step 1 - define our hyper-parameters ie the tuning knobs which define how our model is analyzing certain data
    # how fast our model converges towards the result i.e the line of best fit for our case?
    learning_rate = 0.0001
    # y = mx + b (gradient slope formula)
    initial_b = 0
    initial_m = 0
    # number of iteration// how much are we gonna train our model?
    num_iterations = 1000

    # step 3 - Train our model
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                      compute_error_for_line_given_points(b, m, points)))


# start with mai because this is the meat of the code
if __name__ == '__main__':
    run()  # run stores all of our logic
