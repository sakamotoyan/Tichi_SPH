
import json, xlwt, os

root = os.getcwd()
save_name = "statistics.xls"

methods = ['VFSPH', 'JL21']
frame_start, frame_end = 1, 1050

def read_file(filename):
    with open(filename) as f:
        json_data = json.load(f)
    data = []
    data.append(('t', json_data["timeInSimulation"]))
    data.append(('dt', json_data["timeStep"]))
    iteration1 = json_data["iteration"]
    data.append(('div_iter', iteration1["divergenceFree_iteration"]))
    data.append(('comp_iter', iteration1["incompressible_iteration"]))
    data.append(('sum_iter', iteration1["sum_iteration"]))
    energy=json_data["energy"]
    data.append(('Ek', energy["statistics_kinetic_energy"]))
    data.append(('Eg', energy["statistics_gravity_potential_energy"]))
    data.append(('E', energy["sum_energy"]))
    phase_energy = energy["phase_kinetic_energy"]
    Ek_phase_sum = 0
    for i in range(len(phase_energy)):
        data.append(('Ek_phase' + str(i), phase_energy[i]))
        Ek_phase_sum += phase_energy[i]
    data.append(('Ek_phase_sum', Ek_phase_sum))
    data.append(('Ek_phase+Eg', Ek_phase_sum + energy["statistics_gravity_potential_energy"]))
    statistics=json_data["statistics"]
    data.append(('comp', statistics["volume_compression"]))
    volume_frac = statistics["volume_frac"]
    for i in range(len(volume_frac)):
        data.append(('vol_frac' + str(i), volume_frac[i]))
    data.append(('t_consume', json_data["time_consumption"]))
    return data

book = xlwt.Workbook()
sheet = book.add_sheet('sheet1')
row = 0

for i in range(frame_start, frame_end):
    data = {}

    for method in methods:
        jsonfile = os.path.join(root, method, 'json', "frame" + str(i) + ".json")
        data[method] = read_file(jsonfile)

    a_data = data[methods[0]]
    cnt = len(a_data)
    
    if i == frame_start:
        sheet.write(row, 0, 'frame')
        col = 1
        for j in range(cnt):
            for method in methods:
                sheet.write(row, col, method + '_' + a_data[j][0])
                col += 1
        row += 1

    sheet.write(row, 0, i)
    col = 1
    for j in range(cnt):
        for method in methods:
            sheet.write(row, col, data[method][j][1])
            col += 1
    row += 1

book.save(os.path.join(root, save_name))