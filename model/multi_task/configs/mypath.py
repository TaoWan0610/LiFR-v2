import os


PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_data_root = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT_DIR)), 'data')
db_root = os.environ.get(
    'MTMAMBA_DATA_ROOT',
    os.environ.get('LIFR_DATA_ROOT', default_data_root)
)

db_names = {
    'PASCALContext': 'PASCALContext',
    'NYUD_MT': 'NYUDv2',
    'Cityscapes': 'cityscapes_all',
    'DSEC': 'dsec',
    'M3ED': 'm3ed-spot/moge2',
    'shf_emergy': 'shf_emergy/moge2',
}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)
