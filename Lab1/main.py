import numpy as numpy
import matplotlib.pyplot as plot
import scipy.stats as stats


def random_value_array(array_count):
    array = numpy.zeros(array_count)
    for i in range(array_count):
        random_number = numpy.random.randint(1, 10)
        if random_number not in array:
            array[i] = random_number
    return numpy.sort(array)


def plot_histogram(discrete_values, value_name):
    if value_name == 'X':
        values = [SV[0] for SV in discrete_values]
    else:
        values = [SV[1] for SV in discrete_values]
    plot_x, plot_y = numpy.unique(values, return_counts=True)
    fig = plot.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    rects1 = ax.bar(plot_x, plot_y / len(values), tick_label=plot_x)
    plot.show()


def get_expected_value(discrete_values, elements_count, value_name):
    if value_name == 'X':
        values = [SV[0] for SV in discrete_values]
    else:
        values = [SV[1] for SV in discrete_values]
    return numpy.sum(values) / elements_count


def get_theoretic_value(theoretic_matrix, value, value_name):
    if value_name == 'X':
        values = numpy.sum(theoretic_matrix, axis=0)
    else:
        values = numpy.sum(theoretic_matrix, axis=1)
    return numpy.sum(values * value)


def get_dispersion(discrete_values, value_name):
    if value_name == 'X':
        values = [SV[1] for SV in discrete_values]
    else:
        values = [SV[1] for SV in discrete_values]
    return numpy.var(values, ddof=1)


def get_theoretic_dispersion(theoretic_matrix, value, value_name):
    if value_name == 'X':
        values = numpy.sum(theoretic_matrix, axis=0)
    else:
        values = numpy.sum(theoretic_matrix, axis=1)
    return numpy.sum(values * (value ** 2)) - numpy.sum(values * value) ** 2


def get_covariance(x_value, y_value, matrix, Mx, My):
    covariance = 0
    for i in range(len(y_value)):
        for j in range(len(x_value)):
            a = matrix[i][j]
            covariance = covariance + (x_value[j] * y_value[i] * matrix[i][j])
    covariance -= Mx * My
    return covariance


def get_correlation(covariance, dispersion_x, dispersion_y):
    return covariance / numpy.sqrt(dispersion_x * dispersion_y)


def intervals_of_expected_value(XY, value_name, count_XY):
    if value_name == 'X':
        values = [SV[0] for SV in XY]
    else:
        values = [SV[1] for SV in XY]
    quantile = stats.norm.ppf((1 + 0.95) / 2)
    return (numpy.mean(values) - numpy.sqrt(numpy.var(values, ddof=1) / count_XY) * quantile,
            numpy.mean(values) + numpy.sqrt(numpy.var(values, ddof=1) / count_XY) * quantile)


x_range = int(input('X range: '))
y_range = int(input('Y range: '))
x_values = random_value_array(x_range)
y_values = random_value_array(y_range)
print('X values: ', x_values)
print('Y values: ', y_values)

probability_matrix = numpy.random.dirichlet(numpy.ones(x_range * y_range))
print('Probability matrix:')
k = 0
for i in range(y_range):
    for j in range(x_range):
        print(probability_matrix[i + j + k], end=' ')
    k = k + 1
    print('')

distribution_x = numpy.zeros(x_range)
for l in range(x_range):
    for i in range(y_range):
        distribution_x[l] += probability_matrix[i * x_range + l]
print('Distribution of x : ', distribution_x)

F_x = numpy.zeros(x_range)
to_add = 0
for i in range(x_range):
    to_add += distribution_x[i]
    F_x[i] = to_add
print('Distribution function of x : ', F_x)

print('Distribution function of x of y: ')
F_x_y = numpy.zeros(x_range)
for j in range(x_range):
    to_add = 0
    for i in range(y_range):
        to_add += probability_matrix[j * y_range + i]
        F_x_y[i] = to_add
print(F_x_y)

xy_count = int(input('XY values count: '))
discrete_XY = numpy.zeros((xy_count, 2))
j_saved = 0
for i in range(xy_count):
    discrete_XY[i][0] = numpy.random.choice(x_values)
    discrete_XY[i][1] = numpy.random.choice(y_values)
print('Generated XY values: ', discrete_XY)

print('Empiric matrix:')
empiric_matrix = numpy.zeros((y_range, x_range))
same_XY_array, XY_drop_count = numpy.unique(discrete_XY, return_counts=True, axis=0)
count_XY_value_types = len(same_XY_array)
for i in range(count_XY_value_types):
    x_position = numpy.where(x_values == same_XY_array[i][0])
    y_position = numpy.where(y_values == same_XY_array[i][1])
    empiric_matrix[y_position, x_position] = XY_drop_count[i] / xy_count
print(empiric_matrix)

plot_histogram(discrete_XY, 'X')
plot_histogram(discrete_XY, 'Y')

expected_x = get_expected_value(discrete_XY, xy_count, 'X')
expected_y = get_expected_value(discrete_XY, xy_count, 'Y')
theoretic_x = get_theoretic_value(numpy.hsplit(probability_matrix, y_range), x_values, 'X')
theoretic_y = get_theoretic_value(numpy.hsplit(probability_matrix, y_range), y_values, 'Y')
print('Empiric М[X] = ', expected_x)
print('Theoretic М[X] = ', theoretic_x)
print('Empiric М[Y] = ', expected_y)
print('Theoretic М[Y] = ', theoretic_y)

empiric_Dx = get_dispersion(discrete_XY, 'X')
empiric_Dy = get_dispersion(discrete_XY, 'Y')
theoretic_Dx = get_theoretic_dispersion(numpy.hsplit(probability_matrix, y_range), x_values, 'X')
theoretic_Dy = get_theoretic_dispersion(numpy.hsplit(probability_matrix, y_range), y_values, 'Y')
print('Empiric D[X] = ', empiric_Dx)
print('Theoretic D[X] = ', theoretic_Dx)
print('Empiric D[Y] = ', empiric_Dy)
print('Theoretic D[Y] = ', theoretic_Dy)

empiric_covariance = get_covariance(x_values, y_values, empiric_matrix, expected_x, expected_y)
theoretic_covariance = get_covariance(x_values, y_values, numpy.hsplit(probability_matrix, y_range),
                                      theoretic_x, theoretic_y)
empiric_correlation = get_correlation(empiric_covariance, empiric_Dx, empiric_Dy)
theoretic_correlation = get_correlation(theoretic_covariance, theoretic_Dx, theoretic_Dx)
print('Empiric covariance = ', empiric_covariance)
print('Theoretic covariance = ', theoretic_covariance)
print('Empiric correlation = ', empiric_correlation)
print('Theoretic correlation = ', theoretic_correlation)

print('Interval for М[X] = ', intervals_of_expected_value(discrete_XY, 'X', xy_count))
print('Interval for М[Y] = ', intervals_of_expected_value(discrete_XY, 'Y', xy_count))
