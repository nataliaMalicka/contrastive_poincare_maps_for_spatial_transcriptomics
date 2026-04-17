
#color dict paul taken over from https://github.com/facebookresearch/PoincareMaps
color_dict_paul = {'12Baso': '#0570b0', '13Baso': '#034e7b',
                   '11DC': '#ffff33',
                   '18Eos': '#2CA02C',
                   '1Ery': '#fed976', '2Ery': '#feb24c', '3Ery': '#fd8d3c', '4Ery': '#fc4e2a', '5Ery': '#e31a1c',
                   '6Ery': '#b10026',
                   '9GMP': '#999999', '10GMP': '#4d4d4d',
                   '19Lymph': '#35978f',
                   '7MEP': '#E377C2',
                   '8Mk': '#BCBD22',
                   '14Mo': '#4eb3d3', '15Mo': '#7bccc4',
                   '16Neu': '#6a51a3', '17Neu': '#3f007d',
                   'root': '#000000'}

color_dict_mouse_brain = {'Choroid Plexus': '#023fa5',
                          'Endothelial': '#7d87b9',
                          'Fibroblast': '#bec1d4',
                          'Radial glia': '#9cded6',
                          'White Blood Cells': '#d5eae7',
                          'Neuron_1': '#d6bcc0',
                          'Neuron_2': '#bb7784',
                          'Neuron_3': '#8e063b',
                          'Neuron_4': '#4a6fe3',
                          'Neuron_5': '#8595e1',
                          'Neuron_6': '#b5bbe3',
                          'Neuron_7': '#e6afb9',
                          'Neuron_8': '#e07b91',
                          'Neuron_9': '#d33f6a',
                          'Neuron_10': '#11c638',
                          'Neuron_11': '#8dd593',
                          'Neuron_12': '#c6dec7',
                          'Neuron_13': '#ead3c6',
                          'Neuron_14': '#f0b98d',
                          'Neuron_15': '#ef9708',
                          'Neuron_16': '#0fcfc0',
                          "Neuron" : "lightgray"
                          }

color_dict_mouse_days = {
    'E6.5': 'lightgray',
    'E6.75': 'lightgray',
    'E7.0': 'lightgray',
    'E7.25': 'lightgray',
    'E7.5': 'lightgray',
    'E7.75': 'lightgray',
    'E8.0': 'lightgray',
    'E8.25': 'lightgray',
    'E8.5': 'lightgray',
    'mixed_gastrulation': 'red'
}

color_dict_chicken_days = {
    "D4": 'forestgreen',
    "D7-LV": 'mediumpurple',
    "D7-RV": "mediumorchid",
    "D10-LV": "palevioletred",
    "D10-RV": "hotpink",
    "D14-LV": "royalblue",
    "D14-RV": "lightblue"
}

colors_chicken = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                  '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                  '#e377c2', '#f7b6d2', '#7f7f7f']

labels_chicken = ["Cardiomyocytes-1", "Cardiomyocytes-2", "Immature myocardial cells", "MT-enriched cardiomyocytes",
                  "Vascular endothelial cells", "Endocardial cells", "Valve cells", "Mural cells",
                  "Fibroblast cells", "Epi-mesenchymal cells", "Epi-epithelial cells", "Dendritic cells",
                  "Erythrocytes", "Macrophages", "TMSB4X high cells"]
color_dict_chicken = dict(zip(labels_chicken, colors_chicken))

color_dict_mouse = {"Epiblast": "#635547",
                    "Primitive Streak": "#DABE99",
                    "Caudal epiblast": "#9e6762",

                    "PGC": "#FACB12",

                    "Anterior Primitive Streak": "#c19f70",
                    "Notochord": "#0F4A9C",
                    "Def. endoderm": "#F397C0",
                    "Gut": "#EF5A9D",

                    "Nascent mesoderm": "#C594BF",
                    "Mixed mesoderm": "#DFCDE4",
                    "Intermediate mesoderm": "#139992",
                    "Caudal Mesoderm": "#3F84AA",
                    "Paraxial mesoderm": "#8DB5CE",
                    "Somitic mesoderm": "#005579",
                    "Pharyngeal mesoderm": "#C9EBFB",
                    "Cardiomyocytes": "#B51D8D",
                    "Allantois": "#532C8A",
                    "ExE mesoderm": "#8870ad",
                    "Mesenchyme": "#cc7818",

                    "Haematoendothelial progenitors": "#FBBE92",
                    "Endothelium": "#ff891c",
                    "Blood progenitors 1": "#f9decf",
                    "Blood progenitors 2": "#c9a997",
                    "Erythroid1": "#C72228",
                    "Erythroid2": "#f79083",
                    "Erythroid3": "#EF4E22",

                    "NMP": "#8EC792",

                    "Rostral neurectoderm": "#65A83E",
                    "Caudal neurectoderm": "#354E23",
                    "Neural crest": "#C3C388",
                    "Forebrain/Midbrain/Hindbrain": "#647a4f",
                    "Spinal cord": "#CDE088",

                    "Surface ectoderm": "#f7f79e",

                    "Visceral endoderm": "#F6BFCB",
                    "ExE endoderm": "#7F6874",
                    "ExE ectoderm": "#989898",
                    "Parietal endoderm": "#1A1A1A"}

color_dict_mouse_days2 = {"E6.5": "#D53E4F",
                          "E6.75": "#F46D43",
                          "E7.0": "#FDAE61",
                          "E7.25": "#FEE08B",
                          "E7.5": "#FFFFBF",
                          "E7.75": "#E6F598",
                          "E8.0": "#ABDDA4",
                          "E8.25": "#66C2A5",
                          "E8.5": "#3288BD",
                          "mixed_gastrulation": "#A9A9A9"}

color_dict_mouse_lineage = {"Endodermal lineage": "#EF5A9D",
                            "Heamato-endothelial lineage": "#EF4E22",
                            "others": "#F6BFCB"
                            }
