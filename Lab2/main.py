from math import factorial
from matplotlib import pyplot as plot
import numpy as numpy
import simpy


fig, axs = plot.subplots(2)


class SMO(object):
    def __init__(self, env, num_channel):
        self.env = env
        self.loader = simpy.Resource(env, num_channel)
        self.total_waiting_times = []
        self.queued_list = []
        self.smo_list = []
        self.queued_times = []
        self.refused_total = []

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
            smo.refused_total.append(count_channel_current + len_queque_current)
            beginning_time = env.now
            result = yield current_request | env.process(smo.waiting(callers, v))
            smo.queued_times.append(env.now - beginning_time)
            if current_request in result:
                yield env.process(smo.refusal(callers, service_flow))
                smo.total_waiting_times.append(env.now - beginning_time)
            else:
                smo.total_waiting_times.append(env.now - beginning_time)
        else:
            smo.refused_total.append(places + number_of_channel + 1)
            smo.queued_times.append(0)
            smo.total_waiting_times.append(0)


def process_SMO(env, smo, number_of_channel, applications_flow, service_flow, v, places):
    callers = 0
    while True:
        yield env.timeout(numpy.random.exponential(1/applications_flow))
        env.process(accepting_applications(env, callers, smo, service_flow, v, places, number_of_channel))
        callers = callers + 1


def generate_SMO(number_of_channel, applications_flow, service_flow, v, places, custom_time):
    env = simpy.Environment()
    smo = SMO(env, number_of_channel)
    env.process(process_SMO(env, smo, number_of_channel, applications_flow, service_flow, v, places))
    env.run(until=custom_time)
    return smo.total_waiting_times, smo.queued_list, smo.queued_times, smo.refused_total, smo.smo_list


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


def get_empiric_probabilities(refused_list, queued_list, queued_times, wait_times, smo_list, number_of_channel, places, applications_flow,
                              service_flow):
    refused_array = numpy.array(refused_list)
    P_x = []
    P_queue = []
    for i in range(1, number_of_channel + places + 2):
        P_x.append(len(refused_array[refused_array == i]) / len(refused_array))
        if i > number_of_channel and i <= number_of_channel + places:
            P_queue.append(len(refused_array[refused_array == i]) / len(refused_array))
    ro = applications_flow / service_flow
    P_refused = len(refused_array[refused_array == (number_of_channel + places + 1)]) / len(refused_array)
    final_propabilities = []
    P_queued = sum(P_queue)
    for i, item in enumerate(P_x):
        print(f'Empiric P{i}: {item}')
        final_propabilities.append(item)
    print("Empiric refuse probality: ", P_refused)
    final_propabilities.append(P_refused)
    print("Empiric queue probality: ", P_queued)
    final_propabilities.append(P_queued)
    throughput = 1 - P_refused
    final_propabilities.append(throughput)
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
    axs[0].hist(wait_times, 50)
    axs[0].set_title('Wait times')
    axs[1].hist(queued_list, 50)
    return final_propabilities


def get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v):
    ro = applications_flow / service_flow
    betta = v / service_flow
    final_propabilities = []
    sum_probalities = 0
    p0 = (sum([ro ** i / factorial(i) for i in range(number_of_channel + 1)]) +
          (ro ** number_of_channel / factorial(number_of_channel)) *
          sum([ro ** i / (numpy.prod([number_of_channel + t * betta for t in range(1, i + 1)])) for i in range(1, places + 1)])) ** -1
    print('Theoretical P0:', p0)
    final_propabilities.append(p0)
    sum_probalities += p0
    for i in range(1, number_of_channel + 1):
        px = (ro ** i / factorial(i)) * p0
        sum_probalities += px
        final_propabilities.append(px)
        print(f'Theoretical P{i}: {px}')
    pn = px
    p_queue = px
    for i in range(1, places + 1):
        px = (ro ** (i) / numpy.prod([number_of_channel + t * betta for t in range(1, i + 1)])) * pn
        sum_probalities += px
        if i < places:
            p_queue += px
        print(f'Theoretical P{number_of_channel + i}: {px}')
        final_propabilities.append(px)
    P = px
    print(f'Theoretical refuse probality: {P}')
    final_propabilities.append(P)
    print("Theoretical queue probality: ", p_queue)
    final_propabilities.append(p_queue)
    throughput = 1 - P
    final_propabilities.append(throughput)
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


number_of_channel = 2
applications_flow = 2
service_flow = 1
places = 2
v = 1

total_waiting_times, queued_list, queued_times, refused_total, smo_list = generate_SMO(number_of_channel, applications_flow, service_flow, v, places, 10000)
P_empiric = get_empiric_probabilities(refused_total, queued_list, queued_times, total_waiting_times, smo_list, number_of_channel, places, applications_flow, service_flow)
P_teoretic = get_teoretic_probabilities(number_of_channel, places, applications_flow, service_flow, v)
plot.show()

