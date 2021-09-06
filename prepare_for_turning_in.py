import shutil
import os

dst_folder = 'project-4'


shutil.copyfile('training.py', os.path.join(dst_folder, 'training.py'))
shutil.copyfile('scoring.py', os.path.join(dst_folder, 'scoring.py'))
shutil.copyfile('deployment.py', os.path.join(dst_folder, 'deployment.py'))
shutil.copyfile('ingestion.py', os.path.join(dst_folder, 'ingestion.py'))
shutil.copyfile('diagnostics.py', os.path.join(dst_folder, 'diagnostics.py'))
shutil.copyfile('reporting.py', os.path.join(dst_folder, 'reporting.py'))
shutil.copyfile('app.py', os.path.join(dst_folder, 'app.py'))
shutil.copyfile('apicalls.py', os.path.join(dst_folder, 'apicalls.py'))
shutil.copyfile('fullprocess.py', os.path.join(dst_folder, 'fullprocess.py'))

shutil.copyfile('ingesteddata/ingestedfiles.txt', os.path.join(dst_folder, 'ingestedfiles.txt'))
shutil.copyfile('ingesteddata/finaldata.csv', os.path.join(dst_folder, 'finaldata.csv'))
shutil.copyfile('production_deployment/trainedmodel.pkl', os.path.join(dst_folder, 'trainedmodel.pkl'))
shutil.copyfile('production_deployment/latestscore.txt', os.path.join(dst_folder, 'latestscore.txt'))


shutil.copyfile('practicemodels/confusionmatrix.png', os.path.join(dst_folder, 'confusionmatrix.png'))
# shutil.copyfile('models/confusionmatrix2.png', os.path.join(dst_folder, 'confusionmatrix2.png'))


shutil.copyfile('apireturn.txt', os.path.join(dst_folder, 'apireturn.txt'))
# shutil.copyfile('apireturn2.txt', os.path.join(dst_folder, 'apireturn2.txt'))

shutil.copyfile('crontab.txt', os.path.join(dst_folder, 'crontab.txt'))