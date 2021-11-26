from math import factorial
from matplotlib import pyplot as plot
import numpy as numpy
import simpy


theor_p = []
empir_p = []
empir_throughput = []


class SMO(object):
    def __init__(self, env):
        self.env = env
        self.loader = simpy.Resource(env)
        self.system_state = []

    def refusal(self, applications_flow):
        yield self.env.timeout(1 / applications_flow)


def accepting_applications(env, smo, applications_flow):
    with smo.loader.request() as current_request:
        if current_request.triggered:
            yield env.process(smo.refusal(applications_flow))
            smo.system_state.append(1)
        else:
            smo.system_state.append(2)
            smo.loader.release(current_request)


def process_SMO(env, smo, applications_flow):
    while True:
        yield env.timeout(numpy.random.exponential(1 / applications_flow))
        env.process(accepting_applications(env, smo, applications_flow))


def generate_SMO(applications_flow, custom_time):
    env = simpy.Environment()
    smo = SMO(env)
    env.process(process_SMO(env, smo, applications_flow))
    env.run(until=custom_time)
    return smo.system_state


def get_empiric_probabilities(system_state):
    final_propabilities = []
    P0_values = 0
    for item in system_state:
        if item == 1:
            P0_values = P0_values + 1
    P0 = P0_values / len(system_state)
    print("Theoretical P0: ", P0)
    final_propabilities.append(P0)
    P1 = 1 - P0
    empir_p.append(P0)
    empir_p.append(P1)
    print("Theoretical P1: ", P1)
    final_propabilities.append(P1)
    throughput = P0
    final_propabilities.append(throughput)
    print("Empiric relative throughput: ", throughput)
    absolute_throughput = throughput * applications_flow
    final_propabilities.append(absolute_throughput)
    print("Empiric absolute throughput: ", absolute_throughput)
    empir_throughput.append(throughput)
    empir_throughput.append(absolute_throughput)
    return final_propabilities


def get_teoretic_probabilities(applications_flow):
    final_propabilities = []
    P0 = applications_flow / (applications_flow + applications_flow)
    final_propabilities.append(P0)
    print("Theoretical P0: ", P0)
    theor_p.append(P0)
    P1 = 1 - P0
    final_propabilities.append(P1)
    print("Theoretical P1: ", P1)
    theor_p.append(P1)
    throughput = P0
    print("Theoretical relative throughput: ", throughput)
    final_propabilities.append(throughput)
    absolute_throughput = throughput * applications_flow
    final_propabilities.append(absolute_throughput)
    print("Theoretical absolute throughput : ", absolute_throughput)
    return final_propabilities


def plot_probabilities(refused_number, theoretical_probabilities, interval_count):
    intervals = numpy.array_split(refused_number, interval_count)
    for i in range(0, len(intervals)):
        intervals[i] = numpy.append(intervals[i], intervals[i - 1])
    for i in range(len(theoretical_probabilities)):
        interval_probabilities = []
        for interval in intervals:
            interval_probabilities.append(len(interval[interval == i+1]) / len(interval))
        plot.figure(figsize=(5, 5))
        plot.bar(range(len(interval_probabilities)), interval_probabilities, color='purple')
        plot.title(f"P[{i}]")
        plot.axhline(y=theoretical_probabilities[i], xmin=0, xmax=len(interval_probabilities), color='red')
        plot.show()


def show_difference(empirics, theoretics):
    fig, ax = plot.subplots()
    labels = ["P[0]", 'P[1]', 'throughput', 'abs throughput']
    ax.bar([i - 0.2 for i in range(len(empirics))], empirics, tick_label=labels, width=0.4, color='red',
           label='Empiric')
    ax.bar([i + 0.2 for i in range(len(empirics))], theoretics, width=0.4, color='purple', label='Theoretical')
    ax.set_title('Empiric and theoretic difference')
    fig.set_figwidth(19)
    fig.set_figheight(6)
    plot.legend()
    plot.show()


def show_changes(empirics1, empirics2, empirics3, value_name, values, labels):
    fig, ax = plot.subplots()
    label1 = value_name + " = " + str(values[0])
    ax.bar([i for i in range(len(empirics1))], empirics1, tick_label=labels, width=0.3, color='green',
           label=label1)
    label2 = value_name + " = " + str(values[1])
    ax.bar([i + 0.3 for i in range(len(empirics1))], empirics2, tick_label=labels, width=0.3, color='yellow',
           label=label2)
    label3 = value_name + " = " + str(values[2])
    ax.bar([i + 0.6 for i in range(len(empirics1))], empirics3, tick_label=labels, width=0.3, color='red',
           label=label3)
    ax.set_title('Empiric values changes')
    fig.set_figwidth(19)
    fig.set_figheight(6)
    plot.legend()
    plot.show()


values = [1, 5, 9]

applications_flow = values[0]
system_state = generate_SMO(applications_flow,  10000)
P_empiric = get_empiric_probabilities(system_state)
P_teoretic = get_teoretic_probabilities(applications_flow)
show_difference(P_empiric, P_teoretic)
plot_probabilities(system_state, theor_p, 1000)
empirics1 = []
emp_throughput1 = []
empirics1.append(empir_p[0])
empirics1.append(empir_p[1])
emp_throughput1.append(empir_throughput[0])
emp_throughput1.append(empir_throughput[1])

applications_flow = values[1]
system_state = generate_SMO(applications_flow,  10000)
P_empiric = get_empiric_probabilities(system_state)
P_teoretic = get_teoretic_probabilities(applications_flow)
empirics2 = []
emp_throughput2 = []
empirics2.append(empir_p[2])
empirics2.append(empir_p[3])
emp_throughput2.append(empir_throughput[2])
emp_throughput2.append(empir_throughput[3])

applications_flow = values[2]
system_state = generate_SMO(applications_flow,  10000)
P_empiric = get_empiric_probabilities(system_state)
P_teoretic = get_teoretic_probabilities(applications_flow)
empirics3 = []
emp_throughput3 = []
empirics3.append(empir_p[4])
empirics3.append(empir_p[5])
emp_throughput3.append(empir_throughput[4])
emp_throughput3.append(empir_throughput[5])

labels = ["P[0]", 'P[1]']
show_changes(empirics1, empirics2, empirics3, "Application flow", values, labels)
labels = ["Relative throughput", 'Absolute throughput']
show_changes(emp_throughput1, emp_throughput2, emp_throughput3, "Application flow", values, labels)
