import pydicom

if __name__ == '__main__':
    file_path = '/media/veracrypt1/breast/volunteer_matai/DICOM/DICOM/PA0/ST0/SE6/IM239'
    ds = pydicom.read_file(file_path)
    print(ds)
    patient_id = ds.PatientID
    print("patient_id: ", patient_id)
    study_instance_uid = ds.StudyInstanceUID
    print("study_instance_uid: ", study_instance_uid)
    study_id = ds.StudyID
    print("study_id: ", study_id)
    print(ds['0019', '105a'])
    print(ds['0018', '9516'])
    print(ds['0018', '9517'])



