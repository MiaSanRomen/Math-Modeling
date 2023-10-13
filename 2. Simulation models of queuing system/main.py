from math import factorial
from matplotlib import pyplot as plot
import numpy as numpy
import simpy


theor_p = []
empir_p = []


class SMO(object):
    def __init__(self, env, num_channel):
        self.env = env
        self.loader = simpy.Resource(env, num_channel)
        self.total_waiting_times = []
        self.queued_list = []
        self.smo_list = []
        self.queued_times = []
        self.system_state = []

    def refusal(self, callers, service_flow):
        yield self.env.timeout(numpy.random.exponential(1 / service_flow))

    def waiting(self, callers, v):
        yield self.env.timeout(numpy.random.exponential(1 / v))


def accepting_applications(env, callers, smo, service_flow, v, places, number_of_channel):
    len_queque_total = len(smo.loader.queue)
    count_active_channel_total = smo.loader.count
    with smo.loader.request() as current_request:
        len_queque_current = len(smo.loader.queue)
        count_channel_current = smo.loader.count
        smo.queued_list.append(len_queque_total)
        smo.smo_list.append(len_queque_total + count_active_channel_total)
        if len_queque_current <= places:
            smo.system_state.append(count_channel_current + len_queque_current)
            beginning_time = env.now
            result = yield current_request | env.process(smo.waiting(callers, v))
            smo.queued_times.append(env.now - beginning_time)
            if current_request in result:
                yield env.process(smo.refusal(callers, service_flow))
                smo.total_waiting_times.append(env.now - beginning_time)
            else:
                smo.total_waiting_times.append(env.now - beginning_time)
        else:
            smo.system_state.append(places + number_of_channel + 1)
            smo.queued_times.append(0)
            smo.total_waiting_times.append(0)


def process_SMO(env, smo, number_of_channel, applications_flow, service_flow, v, places):
    callers = 0
    while True:
        yield env.timeout(numpy.random.exponential(1 / applications_flow))
        env.process(accepting_applications(env, callers, smo, service_flow, v, places, number_of_channel))
        callers = callers + 1


def generate_SMO(number_of_channel, applications_flow, service_flow, v, places, custom_time):
    env = simpy.Environment()
    smo = SMO(env, number_of_channel)
    env.process(process_SMO(env, smo, number_of_channel, applications_flow, service_flow, v, places))
    env.run(until=custom_time)
    return smo.total_waiting_times, smo.queued_list, smo.queued_times, smo.system_state, smo.smo_list


def average_applications_count(queued_list):
    average_count = numpy.array(queued_list).mean()
    print("Average applications in queue: ", average_count)
    return average_count


def average_in_SMO(smo_list):
    average_count = numpy.array(smo_list).mean()
    print("Average applications, processing in SMO: ", average_count)
    return average_count


def average_queue_time(queued_times):
    average_queued_time = numpy.array(queued_times).mean()
    print("Average application time elapsed in queue: %s" % (average_queued_time))
    return average_queued_time


def average_smo_time(total_waiting_times):
    average_time = numpy.array(total_waiting_times).mean()
    print("Average application time elapsed in queue: %s " % (average_time))
    return average_time


def get_empiric_probabilities(refused_list, queued_list, queued_times, wait_times, smo_list, number_of_channel, places,
                              applications_flow,
                              service_flow):
    refused_array = numpy.array(refused_list)
    P_x = []
    P_queue = []
    for i in range(1, number_of_channel + places + 2):
        P_x.append(len(refused_array[refused_array == i]) / len(refused_array))
        if i > number_of_channel and i <= number_of_channel + places:
            P_queue.append(len(refused_array[refused_array == i]) / len(refused_array))
    P_refused = len(refused_array[refused_array == (number_of_channel + places + 1)]) / len(refused_array)
    final_propabilities = []
    P_queued = sum(P_queue)
    for i, item in enumerate(P_x):
        print(f'Empiric P{i}: {item}')
        empir_p.append(item)
        final_propabilities.append(item)
    print("Empiric refuse probality: ", P_refused)
    final_propabilities.append(P_refused)
    print("Empiric queue probality: ", P_queued)
    throughput = 1 - P_refused
    print("Empiric throughput: ", throughput)
    absolute_throughput = throughput * applications_flow
    final_propabilities.append(absolute_throughput)
    print("Empiric absolute throughput: ", absolute_throughput)
    n_people_queque = average_applications_count(queued_list)
    final_propabilities.append(n_people_queque)
    av_in_SMO = average_in_SMO(smo_list)
    final_propabilities.append(av_in_SMO)
    n_average = throughput * applications_flow / service_flow
    final_propabilities.append(n_average)
    print("Average number of full channels: ", n_average)
    total_queue = average_applications_count(queued_times)
    final_propabilities.append(total_queue)
    total_smo = average_smo_time(wait_times)
    final_propabilities.append(total_smo)
    return final_propabilities


def get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v):
    ro = applications_flow / service_flow
    betta = v / service_flow
    final_propabilities = []
    sum_probalities = 0
    p0 = (sum([ro ** i / factorial(i) for i in range(number_of_channel + 1)]) +
          (ro ** number_of_channel / factorial(number_of_channel)) *
          sum([ro ** i / (numpy.prod([number_of_channel + t * betta for t in range(1, i + 1)])) for i in
               range(1, places + 1)])) ** -1
    print('Theoretical P0:', p0)
    final_propabilities.append(p0)
    theor_p.append(p0)
    sum_probalities += p0
    for i in range(1, number_of_channel + 1):
        px = (ro ** i / factorial(i)) * p0
        sum_probalities += px
        final_propabilities.append(px)
        print(f'Theoretical P{i}: {px}')
        theor_p.append(px)
    pn = px
    p_queue = px
    for i in range(1, places + 1):
        px = (ro ** (i) / numpy.prod([number_of_channel + t * betta for t in range(1, i + 1)])) * pn
        sum_probalities += px
        if i < places:
            p_queue += px
        print(f'Theoretical P{number_of_channel + i}: {px}')
        theor_p.append(px)
        final_propabilities.append(px)
    P = px
    print(f'Theoretical refuse probality: {P}')
    final_propabilities.append(P)
    print("Theoretical queue probality: ", p_queue)
    throughput = 1 - P
    print("Theoretical throughput: ", throughput)
    absolute_throughput = throughput * applications_flow
    final_propabilities.append(absolute_throughput)
    print("Theoretical absolute throughput : ", absolute_throughput)
    n_people_queque = sum([i * pn * (ro ** i) / numpy.prod([number_of_channel + l * betta for l in range(1, i + 1)]) for
                           i in range(1, places + 1)])
    final_propabilities.append(n_people_queque)
    print("Average applications in queue: ", n_people_queque)
    K_av = sum([index * p0 * (ro ** index) / factorial(index) for index in range(1, number_of_channel + 1)]) + sum(
        [(number_of_channel + index) * pn * ro ** index / numpy.prod(
            numpy.array([number_of_channel + l * betta for l in range(1, index + 1)])) for
         index in range(1, places + 1)])
    final_propabilities.append(K_av)
    print("Average applications processed in SMO : ", K_av)
    n_average = throughput * ro
    final_propabilities.append(n_average)
    print("Average number of full channels: ", n_average)
    time_queue = n_people_queque / applications_flow
    final_propabilities.append(time_queue)
    print("Average application time in queue: ", time_queue)
    time_smo = K_av / applications_flow
    final_propabilities.append(time_smo)
    print("Average application time in SMO: ", time_smo)
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
        red_line = theoretical_probabilities[i]
        plot.axhline(y=theoretical_probabilities[i], xmin=0, xmax=len(interval_probabilities), color='red')
        plot.show()


def show_difference(empirics, theoretics):
    fig, ax = plot.subplots()
    labels = ["P[0]", 'P[1]', 'P[2]', 'P[3]', 'P[4]', 'P refuse', 'abs throughput', 'appl./queue', 'appl./SMO',
              'Full ch.', 'appl. t/queue', 'appl. t/SMO']
    ax.bar([i - 0.2 for i in range(len(empirics))], empirics, tick_label=labels, width=0.4, color='red',
           label='Empiric')
    ax.bar([i + 0.2 for i in range(len(empirics))], theoretics, width=0.4, color='purple', label='Theoretical')
    ax.set_title('Empiric and theoretic difference')
    fig.set_figwidth(19)
    fig.set_figheight(6)
    plot.legend()
    plot.show()


def show_changes(empirics1, empirics2, empirics3, value_name, values):
    fig, ax = plot.subplots()
    labels = ["P[0]", 'P[1]', 'P[2]', 'P[3]', 'P[4]']
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


number_of_channel = 2
service_flow = 1
places = 2
v = 1

values = [1, 4, 7]
applications_flow = values[0]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
show_difference(P_empiric, P_teoretic)
plot_probabilities(system_state, theor_p, 1000)
empirics1 = []
empirics1.append(empir_p[0])
empirics1.append(empir_p[1])
empirics1.append(empir_p[2])
empirics1.append(empir_p[3])
empirics1.append(empir_p[4])

applications_flow = values[1]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics2 = []
empirics2.append(empir_p[5])
empirics2.append(empir_p[6])
empirics2.append(empir_p[7])
empirics2.append(empir_p[8])
empirics2.append(empir_p[9])

applications_flow = values[2]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics3 = []
empirics3.append(empir_p[10])
empirics3.append(empir_p[11])
empirics3.append(empir_p[12])
empirics3.append(empir_p[13])
empirics3.append(empir_p[14])


show_changes(empirics1, empirics2, empirics3, "Application flow", values)

#---------------------------------------------------------------------------------------------------------
applications_flow = 2
empirics1.clear()
empirics3.clear()
empirics2.clear()
empir_p.clear()
values.clear()
values = [1, 4, 7]

service_flow = values[0]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics1 = []
empirics1.append(empir_p[0])
empirics1.append(empir_p[1])
empirics1.append(empir_p[2])
empirics1.append(empir_p[3])
empirics1.append(empir_p[4])


service_flow = values[1]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics2 = []
empirics2.append(empir_p[5])
empirics2.append(empir_p[6])
empirics2.append(empir_p[7])
empirics2.append(empir_p[8])
empirics2.append(empir_p[9])


service_flow = values[2]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics3 = []
empirics3.append(empir_p[10])
empirics3.append(empir_p[11])
empirics3.append(empir_p[12])
empirics3.append(empir_p[13])
empirics3.append(empir_p[14])


show_changes(empirics1, empirics2, empirics3, "Service flow", values)


#---------------------------------------------------------------------------------------------------------
service_flow = 2
empirics1.clear()
empirics3.clear()
empirics2.clear()
empir_p.clear()
values.clear()
values = [2, 3, 4]

v = values[0]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics1 = []
empirics1.append(empir_p[0])
empirics1.append(empir_p[1])
empirics1.append(empir_p[2])
empirics1.append(empir_p[3])
empirics1.append(empir_p[4])


v = values[1]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics2 = []
empirics2.append(empir_p[5])
empirics2.append(empir_p[6])
empirics2.append(empir_p[7])
empirics2.append(empir_p[8])
empirics2.append(empir_p[9])


v = values[2]
total_waiting_times, queued_list, queued_times, system_state, smo_list = generate_SMO(number_of_channel,
                                                                                       applications_flow, service_flow,
                                                                                      v, places, 15000)
P_empiric = get_empiric_probabilities(system_state, queued_list, queued_times, total_waiting_times, smo_list,
                                      number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
empirics3 = []
empirics3.append(empir_p[10])
empirics3.append(empir_p[11])
empirics3.append(empir_p[12])
empirics3.append(empir_p[13])
empirics3.append(empir_p[14])


show_changes(empirics1, empirics2, empirics3, "V", values)
