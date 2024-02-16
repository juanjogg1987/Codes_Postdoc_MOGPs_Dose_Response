import numpy as np
import glob

file_1 = 'Rebuttal_Files_MeasureTimes_MOGP/'  # Replace with the actual path to your text file

time_all_cancers = []
for sel_can in range(0,10):
    file_2 = 'JobsRebuttal_Cancer'+str(sel_can)+'_Train1Cancer_GPyjobs_N_Drugs_GPy_ExactMOGP_ProdKern/'
    #file_path = file_1+file_2+ 'bash0.sh.o8898392' #'bash1.sh.o8898393'

    # Specify the directory and the partial filename pattern
    directory_path = file_1+file_2
    time_all_models = []
    for i in range(0,44):
        partial_filename_pattern = 'bash'+str(i)+'.sh.o*'

        # Use glob to find files matching the pattern in the specified directory
        matching_files = glob.glob(directory_path + partial_filename_pattern)

        # Check if any matching files are found
        if matching_files:
            # Open the first matching file (you may need to handle multiple matches differently)
            file_path = matching_files[0]

            # Open the file
            with open(file_path, 'r') as file:
                # Iterate through each line in the file
                for line in file:
                    # Check if the line contains the target word
                    if 'Runtime:' in line:
                        # Extract the value (assuming the format is consistent)
                        # You may need to adjust this based on the actual format of your data
                        value_start_index = len('Runtime: ')
                        try:
                            value_end_index2 = line.find('h', value_start_index)
                            #print(value_end_index2)
                            hours = line[value_start_index:value_end_index2]
                            #print(value_end_index2)
                            minutes = line[value_end_index2 + 1:value_end_index2 + 3]

                            total_min = float(hours)*60.0 + float(minutes)
                            # Print or use the extracted value
                            print("Hours:", minutes)
                            print("Minutes:", minutes)
                            print("Total Minutes", total_min)
                            time_all_models.append(total_min)
                        except:
                            value_end_index2 = line.find('m', value_start_index)
                            #print(value_end_index2)
                            minutes = line[value_start_index:value_end_index2]
                            #print(value_end_index2)
                            seconds = line[value_end_index2 + 1:value_end_index2 + 3]

                            total_min = float(minutes) + float(seconds) / 60.0
                            # Print or use the extracted value
                            print("Minutes:", minutes)
                            print("Seconds:", seconds)
                            print("Total Minutes", total_min)
                            time_all_models.append(total_min)
        else:
            print("No matching file found.")

    time_all_cancers.append(time_all_models)

dict_cancers = {0: 'GDSC2_10drugs_SKCM_1000FR.csv', 1: 'GDSC2_10drugs_SCLC_1000FR.csv',
                2: 'GDSC2_10drugs_PAAD_1000FR.csv', 3: 'GDSC2_10drugs_OV_1000FR.csv',
                4: 'GDSC2_10drugs_LUAD_1000FR.csv',
                5: 'GDSC2_10drugs_HNSC_1000FR.csv', 6: 'GDSC2_10drugs_ESCA_1000FR.csv',
                7: 'GDSC2_10drugs_COAD_1000FR.csv',
                8: 'GDSC2_10drugs_BRCA_1000FR.csv', 9: 'GDSC2_10drugs_ALL_1000FR.csv'}

import matplotlib.pyplot as plt

mylabels = ['SKCM','SCLC','PAAD','OV','LUAD',
            'HNSC','ESCA','COAD','BRCA','ALL']

dict_Nsize = {'SKCM':456,'SCLC':501,'PAAD':253,'OV':292,'LUAD':537,
            'HNSC':305,'ESCA':319,'COAD':426,'BRCA':441,'ALL':244}

Nsize_list = [dict_Nsize[mylabels[N]] for N in range(10)]
idx_sort = np.argsort(Nsize_list)

#boxplot = plt.boxplot(time_all_cancers,notch=True,labels=mylabels)
sorted_times = [time_all_cancers[i] for i in idx_sort]
sorted_labels = [mylabels[i] for i in idx_sort]
boxplot = plt.boxplot(sorted_times,notch=True,labels=sorted_labels)

for i, box in enumerate(boxplot['boxes']):
    y_position = box.get_ydata()[0]
    x_position = i + 1
    num_observations = dict_Nsize[sorted_labels[i]]
    #plt.text(x_position, 1.1, f'N={num_observations}', ha='center', va='bottom',fontsize=8, fontdict={'weight': 'bold'})
    plt.text(x_position, 1.2, f'N={num_observations}', ha='center', va='bottom', fontsize=9)

plt.xlabel('Cancer type',fontweight='bold', fontsize=14)
plt.ylabel('Minutes',fontweight='bold', fontsize=14)
plt.title('Time required for training the MOGP model',fontweight='bold', fontsize=14)

# Make tick labels bold
for tick_label in plt.gca().get_xticklabels():
    tick_label.set_fontweight('bold')