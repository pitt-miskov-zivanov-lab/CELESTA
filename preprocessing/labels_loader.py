# TODO: check all the labels
import copy

context_types = ['location', 'cell_line', 'cell_type', 'organ', 'disease', 'species']
task_classes = dict()

RELATION = ['Activation', 'Inhibition', 'IncreaseAmount', 'DecreaseAmount', 'GtpActivation']
task_classes['relation'] = RELATION

LOCATION = ['Cytoplasm', 'Cell Membrane', 'Cell Nucleus',
            'Extracellular Matrix', 'Extracellular Space']  # removed 3
task_classes['location'] = LOCATION

CELL_LINE = ['HeLa cell', 'MCF7 cell', '293 cell', 'A549 cell', 'NIH-3T3 cell',
             'Hep G2 cell', 'U-937 cell', 'THP-1 cell', 'LNCAP cell', 'COS-1 cell']
task_classes['cell_line'] = CELL_LINE

CELL_TYPE = ['macrophage', 'endothelial cell', 'fibroblast', 'monocyte', 'fat cell',
             'smooth muscle cell', 'hepatocyte', 'T cell', 'cardiac muscle cell', 'neuron']
task_classes['cell_type'] = CELL_TYPE

ORGAN = ['liver', 'lung', 'skeletal muscle tissue', 'endothelium', 'cardiovascular system',
         'adipose tissue', 'heart', 'cardiovascular system endothelium', 'blood vessel smooth muscle',
         'aorta']
task_classes['organ'] = ORGAN

DISEASE = ['neuroblastoma', 'breast cancer', 'lung cancer', 'atherosclerosis',
           'multiple myeloma', 'leukemia', 'melanoma', 'osteosarcoma', 'non-small cell lung carcinoma']
task_classes['disease'] = DISEASE

SPECIES = ['Homo sapiens', 'Mus musculus', 'Rattus norvegicus']
task_classes['species'] = SPECIES

context_mapping = dict()
tasks = copy.deepcopy(context_types)
tasks.append('relation')
for c in tasks:
    id2label = dict()
    label2id = dict()

    for idx, loc in enumerate(task_classes[c]):
        id2label[idx] = loc
        label2id[loc] = idx

    context_mapping[c] = (id2label, label2id)

cols_values_lst = [LOCATION, CELL_LINE, CELL_TYPE, ORGAN, DISEASE, SPECIES]
num_classes = [len(LOCATION), len(CELL_LINE), len(CELL_TYPE), len(ORGAN), len(DISEASE), len(SPECIES)]

# define labels for tasks
dict_labels = dict(zip(context_types, cols_values_lst))

# augmented data columns
aug_cols = ['umls', 'sr', 'ri', 'rs', 'rd']
