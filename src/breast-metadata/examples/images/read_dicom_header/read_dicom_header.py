import pydicom

if __name__ == '__main__':
    file_path = '/path/to/dicom.dcm'
    ds = pydicom.read_file(file_path)
    print(ds)
    patient_id = ds.PatientID
    print("patient_id: ", patient_id)
    study_instance_uid = ds.StudyInstanceUID
    print("study_instance_uid: ", study_instance_uid)
    study_id = ds.StudyID
    print("study_id: ", study_id)
