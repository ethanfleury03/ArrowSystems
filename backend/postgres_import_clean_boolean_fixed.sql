BEGIN;
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (1, 'admin@example.com', 'Administrator', 'ADMIN', '$2b$12$9O6ixUGzGMP97nvCz1GZyuw0g1I3EcEogJJA1ULfkrynvD3dyDw8.', '2025-11-11 19:52:50.269886', '2025-11-11 19:52:50.269889', NULL, NULL, NULL, '[]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (3, 'ethan@gmail.com', 'ethan', 'ADMIN', '$2b$12$PkcCDTB0zwlU/HHTyka3AeRPYae2FkC5lkC0ojObzCi3HSBGT/EtC', '2025-11-12 17:01:43.324392', '2025-11-12 18:26:17.549509', NULL, NULL, NULL, '[]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (4, 'test@gmail.com', 'test', 'TECHNICIAN', '$2b$12$GfiCMOkNm1UlSGXAmExt3eg.6EWzbZpgC2s9JINmN1TGrdLmq1YCy', '2025-11-12 18:32:14.093820', '2025-11-12 18:32:14.093823', NULL, NULL, NULL, '[]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (5, 'nike@example.com', 'Ethan Fleury', 'CUSTOMER', '$2b$12$/CKvHmOKgGJ7zFHq//QLNeLzrzg4geN147DZNRmFtKn6C2nf/GLqS', '2025-11-12 18:53:17.226607', '2025-11-14 18:36:24.687819', 'Nike', 'Ethan Fleury', '6073731926', '["Any"]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (6, 'api_user', 'api_user', 'TECHNICIAN', '', '2025-11-12 18:54:29.584752', '2025-11-12 18:54:29.584756', NULL, NULL, NULL, '[]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (7, '330@gmail.com', 'temp', 'CUSTOMER', '$2b$12$AQxgQqcvBKMXMzXQFsW59OxpKB7RpseMpaCPyIBkFbYat4aUkhgiO', '2025-11-14 19:46:51.919047', '2025-11-14 19:46:51.919049', '330R', 'temp', 'temp', '["EZCut 350R"]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (8, 'temp@gmail.com', 'test', 'CUSTOMER', '$2b$12$1UPF.nbHJqaYuEvpePXkEO1LJizAdQ0egfWk6Hqi/iG8TLQLKH8O6', '2025-11-14 20:39:57.821919', '2025-11-17 15:01:24.404598', 'test company', 'test', 'test', '["DuraBolt", "EZCut 350R"]');
INSERT INTO users (id, email, name, role, password_hash, created_at, updated_at, company_name, contact_name, contact_phone, machine_models) VALUES (9, 'temps@gmail.com', 'test', 'CUSTOMER', '$2b$12$/U2tvngEcXDcQyrpMqxMEOaOXOtgwSscIp/5DHsgLywHJ.pNdzMPq', '2025-11-18 19:13:12.373849', '2025-11-18 19:13:12.373852', 'temp', 'test', 'test', '["DuraLink"]');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (1, '2800 Series Mini Laser Pro', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (2, 'Duraflex', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (3, 'Anycut', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (4, 'anyCutII', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (5, 'anyCutIII', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (6, 'Anytron AnyJet', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (7, 'ANYTRON Any-002', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (8, 'Digital Die Cutter VR350', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (9, 'DuraLink', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (10, 'DuraBolt', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (11, 'DuraCore', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (12, 'EZCut 330', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (13, 'EZCut 350R', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO machine_models (id, name, created_at, updated_at) VALUES (14, 'GENERAL', '2025-11-18 16:27:40', '2025-11-18 16:27:40');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('186f5200-bd7d-4b4a-83ef-7e2776e4ea4d', '2800 Series Mini Laser Pro User Manual Draft.pdf', '2800 Series Mini Laser Pro', 'COMPLETE', NULL, 'data/original_pdfs_test/2800 Series Mini Laser Pro User Manual Draft.pdf', 3667730, NULL, '2025-11-18 19:44:19.834414', '2025-11-18 19:48:29.774923');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('c5588493-3d7f-486f-b52c-d237960ba4d0', 'Arrow Any-002 Manual.pdf', 'UNKNOWN', 'COMPLETE', NULL, 'data/Arrow Any-002 Manual.pdf', 6502484, NULL, '2025-11-18 20:28:00.604491', '2025-11-18 20:28:00.604494');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('a0835433-d458-4d05-9a46-a52e4866a228', 'print_quality_artefacts_reference_guide.pdf', 'GENERAL', 'COMPLETE', NULL, 'data/print_quality_artefacts_reference_guide.pdf', 8305658, NULL, '2025-11-18 20:28:00.800413', '2025-11-18 20:28:00.800416');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('8533fac3-580c-438f-abe1-87bf77468113', 'anycut_user manual_eng_20150526.docx', 'Anycut', 'COMPLETE', NULL, 'data/anycut_user manual_eng_20150526.docx', 12600965, NULL, '2025-11-18 20:28:01.131489', '2025-11-18 20:28:01.131493');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('de61bd40-3ba2-41e7-868c-20dd5edf516f', 'ANYJET User Guide V1.1.pdf', 'UNKNOWN', 'COMPLETE', NULL, 'data/ANYJET User Guide V1.1.pdf', 6055812, NULL, '2025-11-18 20:28:01.287290', '2025-11-18 20:28:01.287293');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('d006914f-96ff-464b-810a-4542ddc71900', 'EZCut 350R Manual (1).docx.pdf', 'EZCut 350R', 'COMPLETE', NULL, 'data/EZCut 350R Manual (1).docx.pdf', 12442366, NULL, '2025-11-18 20:28:01.673093', '2025-11-18 20:28:01.673097');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('1db314af-5b98-4ac2-9a98-4eef93580dba', 'EZCut 330 User Manual Ver.202106 (1).pdf', 'EZCut 330', 'COMPLETE', NULL, 'data/EZCut 330 User Manual Ver.202106 (1).pdf', 9391988, NULL, '2025-11-18 20:28:00.170289', '2025-11-18 20:28:00.170292');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('ed51bcaf-998b-4775-b6fb-c78aea8b4555', 'anyCUTII User Manual_ver1.0_ENG (1).pdf', 'anyCutII', 'COMPLETE', NULL, 'data/anyCUTII User Manual_ver1.0_ENG (1).pdf', 5882193, NULL, '2025-11-18 20:28:00.216308', '2025-11-18 20:28:00.216313');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('3c3f08cf-402e-4e5d-97a3-845af63c391d', 'anyCUTIII User Manual_ver1.1_ENG_Ana (2).pdf', 'anyCutIII', 'COMPLETE', NULL, 'data/anyCUTIII User Manual_ver1.1_ENG_Ana (2).pdf', 6097620, NULL, '2025-11-18 20:28:00.261991', '2025-11-18 20:28:00.261994');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('85dbe716-dd14-4757-960b-b9cf3fe960db', 'Dura-Printer MCS2.0 Setup and Installation Guide V1.00_21Dec2022.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/Dura-Printer MCS2.0 Setup and Installation Guide V1.00_21Dec2022.pdf', 1511411, NULL, '2025-11-18 20:46:22.226297', '2025-11-18 20:46:22.226300');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('c1c28df4-ba0e-4262-adf2-b681fb8a8ad9', 'DuraFlex Software Release Notes R4.2.3 V1.00_30Apr2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Software Release Notes R4.2.3 V1.00_30Apr2021.pdf', 569234, NULL, '2025-11-18 20:46:22.250637', '2025-11-18 20:46:22.250640');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('368bb3a3-56f2-4475-abdb-b989750db477', 'DuraFlex Electrical Databook and Design Guide V4.03_02Aug2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Electrical Databook and Design Guide V4.03_02Aug2021.pdf', 1624787, NULL, '2025-11-18 20:46:22.321911', '2025-11-18 20:46:22.321915');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('a1b41bc9-74f0-4d86-bc5b-91323d47882b', 'DuraFlex Job Submission Library Guide V1.01_30Apr2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Job Submission Library Guide V1.01_30Apr2021.pdf', 643527, NULL, '2025-11-18 20:46:22.367981', '2025-11-18 20:46:22.367984');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('0c08f748-3e03-4ace-8be6-05980702b93f', 'DuraFlex_MagV2 transition Overview_30Jan2025_R1.0.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex_MagV2 transition Overview_30Jan2025_R1.0.pdf', 1537121, NULL, '2025-11-18 20:46:22.393988', '2025-11-18 20:46:22.393992');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('a12ed6c7-4a2e-474a-a164-f6510c1691a8', 'DuraFlex Installation and Commissioning Guide V5.01_06Apr2023.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Installation and Commissioning Guide V5.01_06Apr2023.pdf', 11923762, NULL, '2025-11-18 20:46:22.454466', '2025-11-18 20:46:22.454469');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('7fb41cad-5e00-4331-b09e-30bec341e983', 'DuraFlex Printhead Storage and Shipping Guide V1.00_01Nov2022 (3).pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Printhead Storage and Shipping Guide V1.00_01Nov2022 (3).pdf', 1179011, NULL, '2025-11-18 20:46:22.611144', '2025-11-18 20:46:22.611146');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('1780db43-91e4-4c06-acc5-eba390d13c0b', 'DuraFlex Troubleshooting Guide V4.05_30May2022.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Troubleshooting Guide V4.05_30May2022.pdf', 3844150, NULL, '2025-11-18 20:46:22.650672', '2025-11-18 20:46:22.650674');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('2f2e7f57-55ec-4ec0-9175-ed0eea9a08ab', 'DuraBolt DuraCore Software Release Notes R3.3.0_V1.01_05Sep2025.pdf', 'DuraBolt', 'COMPLETE', NULL, 'data/DuraBolt DuraCore Software Release Notes R3.3.0_V1.01_05Sep2025.pdf', 1545515, NULL, '2025-11-18 20:46:22.670447', '2025-11-18 20:46:22.670450');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('816a2318-0067-4ff8-b24d-a309a8061a8a', 'DuraFlex JIT Printing Guide V1.00_24May24.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex JIT Printing Guide V1.00_24May24.pdf', 414024, NULL, '2025-11-18 20:46:22.696112', '2025-11-18 20:46:22.696115');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('465c499c-2a77-4ce4-b435-f790a0af58c5', 'DuraFlex TB0011 PH Hydration Settings 01Nov2022.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex TB0011 PH Hydration Settings 01Nov2022.pdf', 580598, NULL, '2025-11-18 20:46:22.739307', '2025-11-18 20:46:22.739311');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('334fcbd3-db37-4bf8-969d-f8952b8722fc', 'DuraFlex Software Release Notes R5.3.3 V1.01_26Aug2025.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Software Release Notes R5.3.3 V1.01_26Aug2025.pdf', 465042, NULL, '2025-11-18 20:46:22.766199', '2025-11-18 20:46:22.766203');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('843adeb0-7808-400e-82d2-d1bf687b43f1', 'DuraFlex_Command Line Interface-Supplimental_V1.0_29April2025.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex_Command Line Interface-Supplimental_V1.0_29April2025.pdf', 753495, NULL, '2025-11-18 20:46:22.800643', '2025-11-18 20:46:22.800647');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('7040b009-f45e-4205-ad13-eea6eff31d8a', 'EF-G3-B Specification sheet  .pdf', 'GENERAL', 'COMPLETE', NULL, 'data/EF-G3-B Specification sheet  .pdf', 415115, NULL, '2025-11-18 20:46:22.823164', '2025-11-18 20:46:22.823166');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('5ef2b46b-6d72-450f-b5c4-2ffba25c8755', 'DuraFlex Software Databook and Design Guide R3.1.1_V1.00_09Jun2020.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Software Databook and Design Guide R3.1.1_V1.00_09Jun2020.pdf', 1170954, NULL, '2025-11-18 20:46:22.884159', '2025-11-18 20:46:22.884161');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('9c343aa4-69e5-4c4e-8d70-2c42d176c5f4', 'Digital_die_cutter_VR350_User_Guide[1].pdf', 'Digital Die Cutter VR350', 'COMPLETE', NULL, 'data/Digital_die_cutter_VR350_User_Guide[1].pdf', 7564782, NULL, '2025-11-18 20:46:22.964581', '2025-11-18 20:46:22.964583');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('e43e5275-1922-4e7a-bab4-5fe9bfd82787', 'DuraCore 1x1 Print Engine Installation Guide_v2.0_18Aug2025.pdf', 'DuraCore', 'COMPLETE', NULL, 'data/DuraCore 1x1 Print Engine Installation Guide_v2.0_18Aug2025.pdf', 21833966, NULL, '2025-11-18 20:46:23.020397', '2025-11-18 20:46:23.020400');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('e0ad732e-eae2-427e-a822-0bf095925226', 'DuraFlex - Service and Support_PPU Procedure - V1.0.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex - Service and Support_PPU Procedure - V1.0.pdf', 1728580, NULL, '2025-11-18 20:46:23.046933', '2025-11-18 20:46:23.046936');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('1bba4b81-f2ed-4feb-8c54-ff9abcc20289', 'DuraFlex TB0007 Inline Degasser Installation 07Dec2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex TB0007 Inline Degasser Installation 07Dec2021.pdf', 863113, NULL, '2025-11-18 20:46:23.071438', '2025-11-18 20:46:23.071441');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('5a5a5d2e-eba6-4ab0-a7b6-54031aa1f24a', 'DuraFlex Spare Parts List V1.03_17April2025.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Spare Parts List V1.03_17April2025.pdf', 1295251, NULL, '2025-11-18 20:46:23.094765', '2025-11-18 20:46:23.094767');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('38ed9326-6ab0-4849-a08f-5dffafcc547f', 'DuraCore Operating and Troubleshooting 01May2025.pdf', 'DuraCore', 'COMPLETE', NULL, 'data/DuraCore Operating and Troubleshooting 01May2025.pdf', 43917042, NULL, '2025-11-18 20:46:23.121222', '2025-11-18 20:46:23.121226');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('b1a0e5d7-c342-4b09-bbf1-ca824efaa5b6', 'DuraFlex System Requirements _ Xitron Support.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex System Requirements _ Xitron Support.pdf', 58039, NULL, '2025-11-18 20:46:23.142893', '2025-11-18 20:46:23.142895');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('7d52ccb3-cea8-4499-aa1d-828f77d8ec33', 'DuraBolt Installation Guide_v6.0_18Aug2025.pdf', 'DuraBolt', 'COMPLETE', NULL, 'data/DuraBolt Installation Guide_v6.0_18Aug2025.pdf', 22256822, NULL, '2025-11-18 20:46:23.224120', '2025-11-18 20:46:23.224122');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('d803fdf7-713f-401e-b87a-67c768e38348', 'DuraFlex Software Release Notes R5.0.2 V1.00_20Aug2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Software Release Notes R5.0.2 V1.00_20Aug2021.pdf', 544495, NULL, '2025-11-18 20:46:23.245449', '2025-11-18 20:46:23.245451');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('a2c23e73-e3c1-41bf-af0b-2620057a7754', 'DuraFlex Printhead Cradle Repair Guide_V1.1_03Oct2025.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Printhead Cradle Repair Guide_V1.1_03Oct2025.pdf', 9075898, NULL, '2025-11-18 20:46:23.282994', '2025-11-18 20:46:23.282996');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('bb24c7b5-2d22-4826-872a-dfeb8ef2eceb', 'DuraFlex TB0008 WIMM Alternate Connection 22Dec2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex TB0008 WIMM Alternate Connection 22Dec2021.pdf', 396203, NULL, '2025-11-18 20:46:23.304536', '2025-11-18 20:46:23.304538');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('db2e1cbe-56a4-44ab-897e-dbb4c81f4f13', 'DuraFlex Software Release Notes R5.2.2 V1.01_29Mar2022.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Software Release Notes R5.2.2 V1.01_29Mar2022.pdf', 781516, NULL, '2025-11-18 20:46:23.331965', '2025-11-18 20:46:23.331968');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('dfda8f8d-8bd5-4b85-a2b3-cb827327c572', 'Connecting to a DuraFlex printer.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/Connecting to a DuraFlex printer.pdf', 2346098, NULL, '2025-11-18 20:46:23.352135', '2025-11-18 20:46:23.352139');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('7fa5a704-8fde-4b36-942b-1073506028bb', 'DuraFlex Operations Guide V4.06_21Sep2022.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Operations Guide V4.06_21Sep2022.pdf', 3198433, NULL, '2025-11-18 20:46:23.402393', '2025-11-18 20:46:23.402396');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('396fe5de-9caf-4eb7-9d65-4132d66dd261', 'Hydration Control.pdf', 'GENERAL', 'COMPLETE', NULL, 'data/Hydration Control.pdf', 142544, NULL, '2025-11-18 20:46:23.420549', '2025-11-18 20:46:23.420551');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('e985436b-60fb-483a-9abc-c7da8c389580', 'DuraFlex Mechanical and Fluidics Databook and Design Guide V4.02_15Jul2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Mechanical and Fluidics Databook and Design Guide V4.02_15Jul2021.pdf', 7754099, NULL, '2025-11-18 20:46:23.444138', '2025-11-18 20:46:23.444141');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('e0404431-299d-4b0f-858f-39372b5d4c51', 'PC Configuration.pdf', 'GENERAL', 'COMPLETE', NULL, 'data/PC Configuration.pdf', 876888, NULL, '2025-11-18 20:46:23.464604', '2025-11-18 20:46:23.464606');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('85486eea-6a0d-4f47-8172-d9a3932343f7', '07.01_D-Flex_Alternate Temperature Regulation Configuration_V1.2_04Oct2024.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/07.01_D-Flex_Alternate Temperature Regulation Configuration_V1.2_04Oct2024.pdf', 245087, NULL, '2025-11-18 20:46:23.484289', '2025-11-18 20:46:23.484291');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('134d9a54-8cdd-4935-aa7d-72b254081871', 'DuraFlex Dehydration Management Summary_R1.0.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Dehydration Management Summary_R1.0.pdf', 185523, NULL, '2025-11-18 20:46:23.515489', '2025-11-18 20:46:23.515492');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('b08efa5e-1e30-4dcf-b0a3-31ec2105d0ac', 'EZCut 330 Troubleshooting.pdf', 'EZCut 330', 'COMPLETE', NULL, 'data/EZCut 330 Troubleshooting.pdf', 2445729, NULL, '2025-11-18 20:46:23.547629', '2025-11-18 20:46:23.547631');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('e6b25cd0-0ff7-438b-8f9d-05137a17ddb6', 'DuraFlex Declog Options Summary_R1.0.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Declog Options Summary_R1.0.pdf', 502324, NULL, '2025-11-18 20:46:23.576328', '2025-11-18 20:46:23.576331');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('6b3dad13-44bf-45f5-9acd-b44d8ad6fd11', 'OperationManualEFSmallSeries_en.pdf', 'GENERAL', 'COMPLETE', NULL, 'data/OperationManualEFSmallSeries_en.pdf', 922026, NULL, '2025-11-18 20:46:23.607206', '2025-11-18 20:46:23.607210');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('d2c8d2a8-9e29-4609-9090-ce7fed87fcda', 'DuraFlex Demo GUI User Guide V4.01_03Jun2021.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Demo GUI User Guide V4.01_03Jun2021.pdf', 1834950, NULL, '2025-11-18 20:46:23.649086', '2025-11-18 20:46:23.649087');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('4cf987b9-5aa8-4ee8-8a1f-4910ce19d35b', 'EZCut 330_How to make the cutting  file.pdf', 'EZCut 330', 'COMPLETE', NULL, 'data/EZCut 330_How to make the cutting  file.pdf', 323702, NULL, '2025-11-18 20:46:23.668935', '2025-11-18 20:46:23.668938');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('0462b0fb-a740-4cba-97ee-2e1dab8b1d58', 'DuraFlex TB0002 Printhead Identification and System Compatibility 20Nov2020.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex TB0002 Printhead Identification and System Compatibility 20Nov2020.pdf', 317265, NULL, '2025-11-18 20:46:23.691536', '2025-11-18 20:46:23.691538');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('8b38f09d-1c90-4686-8286-c798ea8aedc5', 'EZCut 330_How to set tension.pdf', 'EZCut 330', 'COMPLETE', NULL, 'data/EZCut 330_How to set tension.pdf', 1912262, NULL, '2025-11-18 20:46:23.711484', '2025-11-18 20:46:23.711501');
INSERT INTO document_ingestion_metadata (id, filename, machine_model, status, description, file_path, file_size_bytes, error_message, created_at, updated_at) VALUES ('abd91fb1-cfd4-4af0-98cf-c2de61f908fa', 'DuraFlex Service and Repair Guide V2.00_12May2023.pdf', 'Duraflex', 'COMPLETE', NULL, 'data/DuraFlex Service and Repair Guide V2.00_12May2023.pdf', 23648286, NULL, '2025-11-18 20:46:23.781186', '2025-11-18 20:46:23.781189');
INSERT INTO alembic_version (version_num) VALUES ('003_ingestion_phase1');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (1, '2025-11-14 20:12:30.553745', 'warning', 'user_login_failed', 'ethan@gmail.com', NULL, '172.18.0.3', '{"reason": "invalid_credentials"}', '2c6c360f-d7ff-4c76-9f89-8a30fb545f55');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (2, '2025-11-14 20:12:33.826902', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '89411de5-53bb-4f71-817f-a3177b50486b');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (3, '2025-11-14 20:12:37.762898', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'a699bbee-ae4f-4d14-bb38-715d11667e5c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (4, '2025-11-14 20:17:19.211841', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '0f00faeb-439c-4502-b581-ebbb36ffd006');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (5, '2025-11-14 20:22:23.354769', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'aeece08a-62ce-467a-9f01-5a4fab927477');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (6, '2025-11-14 20:27:06.600358', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'a2e725d1-42f3-4879-b8e8-51ba263c5d63');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (7, '2025-11-14 20:32:20.669527', 'warning', 'user_login_failed', 'ethan@gmail.com', NULL, '172.18.0.3', '{"reason": "invalid_credentials"}', '52834e47-3310-48c5-ada8-7f0630593231');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (8, '2025-11-14 20:32:25.145352', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'b1f76b41-73ae-4bed-a3a2-7efc99a68e8b');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (9, '2025-11-14 20:37:58.632398', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '0618dad4-e3c0-44d0-aba5-71be15910ec1');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (10, '2025-11-14 20:39:57.836639', 'info', 'admin_created_user', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"created_user_email": "temp@gmail.com", "created_user_role": "CUSTOMER", "created_user_id": "8", "machine_models": ["EZCut 350R"]}', '4f59dccf-4b4c-4151-93ca-fd4a814d726c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (11, '2025-11-14 20:40:08.481685', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '886078a7-f993-415c-aea1-20e030594a38');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (12, '2025-11-14 20:41:49.370484', 'info', 'rag_query', NULL, 'ADMIN', NULL, '{"query": "Network not connected", "session_id": "session_1763152844326", "chunks_retrieved": 6, "response_time_ms": 65043, "intent_type": "troubleshooting"}', '5064088a-58fa-4b0f-a2f5-db43701fb688');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (13, '2025-11-14 20:43:17.334572', 'info', 'rag_query', NULL, 'ADMIN', NULL, '{"query": "350R Network not connected", "session_id": "session_1763152924370", "chunks_retrieved": 3, "response_time_ms": 72964, "intent_type": "troubleshooting"}', 'bdda9b7d-f30b-453a-9274-0d21e39f93fd');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (14, '2025-11-14 20:49:17.592102', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', 'de71c34b-1679-4399-8d0f-1b62ec744a20');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (15, '2025-11-14 20:50:40.957235', 'info', 'rag_query', NULL, 'ADMIN', NULL, '{"query": "network connection error ", "session_id": "session_1763153372891", "chunks_retrieved": 4, "response_time_ms": 68065, "intent_type": "troubleshooting"}', '8c955a74-5828-41bc-a078-787bff221640');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (16, '2025-11-14 21:00:29.052781', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', 'c97ec370-266d-419a-802e-d0e2d4a138b4');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (17, '2025-11-14 21:01:23.903224', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "test", "session_id": "session_1763154032053", "chunks_retrieved": 3, "response_time_ms": 51849, "intent_type": "definition"}', 'fe73b36b-fbbc-48e4-9fc6-ab5ee08e80af');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (18, '2025-11-14 21:03:01.678033', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "How do i set up the Lamination?", "session_id": "session_1763154095025", "chunks_retrieved": 1, "response_time_ms": 86652, "intent_type": "reasoning"}', 'd27cf958-269f-42c2-98f7-e7943ed564af');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (19, '2025-11-14 21:13:57.873200', 'warning', 'user_login_failed', 'ethan@gmail.com', NULL, '172.18.0.3', '{"reason": "invalid_credentials"}', '61d17026-1bf6-4234-a8bf-fbebdb44a964');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (20, '2025-11-14 21:14:02.902564', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '6bba0369-bb77-4c14-8a19-b424a242aca1');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (21, '2025-11-14 21:18:15.049318', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '8a2236bb-5be3-4368-b726-d13b9cc9ad89');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (22, '2025-11-14 21:22:54.312476', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '367ce561-252c-4175-8558-80a8468208b2');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (23, '2025-11-14 21:27:52.495025', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '2d771755-4f7c-47aa-84ef-f97aeffbd883');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (24, '2025-11-14 21:33:12.753317', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '6383e608-bdfe-4d68-bd52-42fce0b7b31f');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (25, '2025-11-14 21:39:08.663543', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '5386fa6d-4025-4ce3-b1c9-fd80b71b9400');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (26, '2025-11-14 21:39:15.494219', 'info', 'test_event', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"test": true, "admin": "ethan@gmail.com"}', 'b836ab09-06e3-45dd-ac8e-c66771951859');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (27, '2025-11-14 21:44:19.359438', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '47a7f2b4-6657-41a9-b989-91a63926322c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (28, '2025-11-14 21:44:53.693179', 'info', 'admin_updated_user', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"updated_user_id": "8", "updated_user_email": "temp@gmail.com", "role_changed": false, "machines_changed": false, "old_machines": ["EZCut 350R"], "new_machines": null}', 'de1a72ae-49da-4a86-8111-7344c5cc5944');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (29, '2025-11-17 14:44:56.283952', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'f4b2711b-3c7d-4c59-ab0a-6b3d802fd515');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (30, '2025-11-17 14:45:46.546527', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '95884f6d-a8a4-44e8-b1dd-76a72f1294ac');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (31, '2025-11-17 14:56:20.403828', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '59c39299-c710-449c-8abe-98f4554bb408');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (32, '2025-11-17 15:00:59.478987', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '23c6149c-96c8-46fd-9e45-4f1936bc7540');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (33, '2025-11-17 15:01:13.459095', 'warning', 'user_login_failed', 'ethan@gmail.com', NULL, '172.18.0.3', '{"reason": "invalid_credentials"}', 'be1a980f-3340-43e1-ab51-51d1a3524433');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (34, '2025-11-17 15:01:17.927220', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'd0c87fef-097e-49c5-b64f-271770f8b8fa');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (35, '2025-11-17 15:01:24.416934', 'info', 'admin_updated_user', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"updated_user_id": "8", "updated_user_email": "temp@gmail.com", "role_changed": false, "machines_changed": true, "old_machines": ["EZCut 350R"], "new_machines": ["DuraBolt", "EZCut 350R"]}', '7b204241-790d-403b-824f-840c37814b07');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (36, '2025-11-17 15:01:33.384718', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '0cdbf3e7-d5b7-4db5-a3f6-e47c330f9cf7');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (37, '2025-11-17 15:02:36.051289', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '554c3c20-22aa-4fa3-bb82-f64cf6f9d4bf');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (38, '2025-11-17 15:10:15.396187', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '9f2db1c9-36c6-4641-95f1-eb39d3d0cce6');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (39, '2025-11-17 15:11:59.689750', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "What are the duraflex system requirements?", "session_id": "session_1763392270267", "chunks_retrieved": 2, "response_time_ms": 49421, "intent_type": "lookup"}', '0509ec33-4c30-434d-9d9b-46c90476e709');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (40, '2025-11-17 15:15:50.412272', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '96bcf955-af77-4b8f-b56b-e72b1b0f6e2c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (41, '2025-11-17 15:17:07.777251', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "What are the duraflex system requirements?", "session_id": "session_1763392560429", "chunks_retrieved": 3, "response_time_ms": 67347, "intent_type": "lookup"}', '088c16f0-74c3-46e5-b7ff-78cd710872aa');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (42, '2025-11-17 15:17:27.520487', 'info', 'user_feedback', 'temp@gmail.com', 'CUSTOMER', '172.18.0.1', '{"is_helpful": true, "query": "What are the duraflex system requirements?", "intent_type": "lookup", "confidence": 0.7328941830485478}', '7d1fe914-af80-492c-993a-88bb49478c89');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (43, '2025-11-17 15:17:30.342772', 'info', 'user_feedback', 'temp@gmail.com', 'CUSTOMER', '172.18.0.1', '{"is_helpful": false, "query": "What are the duraflex system requirements?", "intent_type": "lookup", "confidence": 0.7328941830485478}', '06b64a2a-04eb-48c4-9a4d-12380b38632e');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (44, '2025-11-17 15:20:38.810332', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "How do I Adjust the blade depth on the 350R?", "session_id": "session_1763392641331", "chunks_retrieved": 1, "response_time_ms": 197478, "intent_type": "reasoning"}', '9ae76049-238f-4818-9476-fd992a0c8025');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (45, '2025-11-17 15:23:57.604667', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "How many mm does each rotation of the knob cause?", "session_id": "session_1763392975480", "chunks_retrieved": 3, "response_time_ms": 62123, "intent_type": "lookup"}', '5d2278de-a3b4-4e83-b29f-d698c243b862');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (46, '2025-11-17 15:26:31.387143', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "are you sure its not .5 mm", "session_id": "session_1763393126487", "chunks_retrieved": 3, "response_time_ms": 64898, "intent_type": "lookup"}', '3f7e9cdd-d8d9-4bff-9edc-0d2c878009ab');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (47, '2025-11-17 15:42:00.709071', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', 'e8831ac2-3727-4d2f-b8c9-379f0f3b55be');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (48, '2025-11-17 15:43:10.511896', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "2", "session_id": "session_1763394144401", "chunks_retrieved": 2, "response_time_ms": 46110, "intent_type": "lookup"}', 'e20856a6-d888-4b24-8616-f8cb7b18ecf2');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (49, '2025-11-17 15:50:56.367526', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '164b2cce-1652-4402-ba25-b2adfe03149d');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (50, '2025-11-17 15:53:26.555519', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "How to check and adjust the verticality on the 350R?", "session_id": "session_1763394708764", "chunks_retrieved": 4, "response_time_ms": 97790, "intent_type": "reasoning"}', 'ada7ee00-ce94-4c34-a7f7-373f40363fbc');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (51, '2025-11-17 17:22:16.976337', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', 'bebf6f30-862d-4773-84b6-66bedd55140b');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (52, '2025-11-17 17:24:22.213796', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "Tell me about the Logs Page on durabolt?", "session_id": "session_1763400196043", "chunks_retrieved": 2, "response_time_ms": 66169, "intent_type": "definition"}', '353ac73e-e825-4a5a-a5d5-ea7891245b3d');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (53, '2025-11-17 17:28:35.043814', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', '006201dd-5857-47b4-88bc-ec2b3ca1c703');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (54, '2025-11-17 17:41:32.370559', 'info', 'user_login', 'temp@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "8"}', 'b3a32fa7-13af-4178-a473-2295095cfa2c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (55, '2025-11-17 17:43:01.641555', 'info', 'rag_query', 'temp@gmail.com', 'CUSTOMER', NULL, '{"query": "How do i make the cut file?", "session_id": "session_1763401315398", "chunks_retrieved": 1, "response_time_ms": 66242, "intent_type": "reasoning"}', '08a71afc-cc8f-4b2c-a0d5-32bbfe2f8d5a');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (56, '2025-11-18 18:10:47.642691', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '291a1bc5-7f44-4f6b-9856-9d4b44d39fc3');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (57, '2025-11-18 18:59:25.578515', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '81fd4eba-c9c2-47bb-bc77-005fa815b645');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (58, '2025-11-18 19:02:42.808288', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '76fc33f9-41a0-40d1-b5f1-c963514cd08c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (59, '2025-11-18 19:04:18.764282', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "DuraBolt"}', 'f3d03c9b-ff54-4c5f-9b5e-fe123284f266');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (60, '2025-11-18 19:04:18.787677', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "DuraBolt", "metadata_id": "3af31cea-bf2e-4578-93f6-2c2e9aef4a94", "status": "PENDING_INGESTION"}', 'f3d03c9b-ff54-4c5f-9b5e-fe123284f266');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (61, '2025-11-18 19:04:48.149897', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "DuraBolt"}', '1f9ac412-9812-49ab-9189-84c409113f7e');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (62, '2025-11-18 19:04:48.169813', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST_20251118_190448.pdf", "machine_model": "DuraBolt", "metadata_id": "69df43c0-8331-4ef2-a1ca-1100bb8ccc00", "status": "PENDING_INGESTION"}', '1f9ac412-9812-49ab-9189-84c409113f7e');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (63, '2025-11-18 19:07:15.835290', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'f6b1b081-e696-412c-beb4-52125c8dd222');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (64, '2025-11-18 19:09:14.359964', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "DuraCore"}', 'd9db31bf-80fa-4596-8b81-92cdae2fe4f5');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (65, '2025-11-18 19:09:14.377970', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST_20251118_190914.pdf", "machine_model": "DuraCore", "metadata_id": "cf199594-e3dd-4f14-9815-ba3de24d8b54", "status": "PENDING_INGESTION"}', 'd9db31bf-80fa-4596-8b81-92cdae2fe4f5');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (66, '2025-11-18 19:09:28.247888', 'info', 'document_metadata_updated', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "updates": {"requires_admin_review": false, "machine_model": ["DuraCore"]}}', '28d9cc73-6c2f-470f-8090-b306266836ba');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (67, '2025-11-18 19:12:40.217086', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "DuraLink"}', '936b97eb-36d7-457f-ab27-655bff033cf3');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (68, '2025-11-18 19:12:40.241455', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST_20251118_191240.pdf", "machine_model": "DuraLink", "metadata_id": "173d527f-354d-4c4c-9ba9-603b1939a437", "status": "PENDING_INGESTION"}', '936b97eb-36d7-457f-ab27-655bff033cf3');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (69, '2025-11-18 19:13:12.379014', 'info', 'admin_created_user', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"created_user_email": "temps@gmail.com", "created_user_role": "CUSTOMER", "created_user_id": "9", "machine_models": ["DuraLink"]}', 'a9711e6a-7af3-42d8-9ddf-f300da8bc316');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (70, '2025-11-18 19:13:27.000543', 'info', 'user_login', 'temps@gmail.com', 'CUSTOMER', '172.18.0.3', '{"user_id": "9"}', 'a718f5c0-da4f-45e1-bee2-3a253b6a2f9d');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (71, '2025-11-18 19:17:31.267558', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '8d6adfab-5236-40b2-aeec-0dac2bf223a3');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (72, '2025-11-18 19:17:43.134003', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "is_active": false, "status": "disabled"}', '3c7b042a-2f5f-4586-acb4-2315fc827677');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (73, '2025-11-18 19:17:44.004734', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "is_active": true, "status": "enabled"}', '040fd135-fe48-4988-9d5a-391f53f634a1');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (74, '2025-11-18 19:17:46.065988', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "is_active": false, "status": "disabled"}', '37921746-0e82-4544-a61e-ca8cbe46318e');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (75, '2025-11-18 19:17:47.679522', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "is_active": true, "status": "enabled"}', '2868e74c-6cfd-48e2-b50b-2d0f6e5a2575');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (76, '2025-11-18 19:17:54.460569', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "2800 Series Mini Laser Pro"}', 'c2cde1b5-d0ea-45fe-bb98-3676b8be536c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (77, '2025-11-18 19:17:54.491044', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST_20251118_191754.pdf", "machine_model": "2800 Series Mini Laser Pro", "metadata_id": "cb9eab94-247f-4264-be89-07cfa01e24ae", "status": "PENDING_INGESTION"}', 'c2cde1b5-d0ea-45fe-bb98-3676b8be536c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (78, '2025-11-18 19:20:13.786930', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "Duraflex"}', 'bdb6cad6-59eb-4ebb-a625-82c10e9da8e6');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (79, '2025-11-18 19:20:13.824102', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST_20251118_192013.pdf", "machine_model": "Duraflex", "metadata_id": "a12da2ff-23bc-4218-a4da-8c08e30b6d93", "status": "PENDING_INGESTION"}', 'bdb6cad6-59eb-4ebb-a625-82c10e9da8e6');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (80, '2025-11-18 19:20:30.574751', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "Arrow Any-002 Manual.pdf", "machine_model": "anyCutIII"}', 'f9d0319f-1e98-4612-9d32-2a5c6c703e81');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (81, '2025-11-18 19:20:30.629008', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "Arrow Any-002 Manual.pdf", "machine_model": "anyCutIII", "metadata_id": "c77e2246-7225-4dcd-b7f4-865b1f561b28", "status": "PENDING_INGESTION"}', 'f9d0319f-1e98-4612-9d32-2a5c6c703e81');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (82, '2025-11-18 19:25:32.832117', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '61055c99-1586-4b46-b935-a7744861d6a7');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (83, '2025-11-18 19:26:31.369834', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST.pdf", "machine_model": "Anycut"}', 'c422dfe8-eeb8-4272-af31-1b3d4d291d09');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (84, '2025-11-18 19:26:31.396438', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "DURABOLTTEST_20251118_192631.pdf", "machine_model": "Anycut", "metadata_id": "0765330f-a2e0-4679-b56d-955c56676f40", "status": "PENDING_INGESTION"}', 'c422dfe8-eeb8-4272-af31-1b3d4d291d09');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (85, '2025-11-18 19:30:46.323297', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '2d825d64-a485-4f01-8344-1443f807903d');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (86, '2025-11-18 19:33:38.618437', 'info', 'test_mode_cleared', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"deleted_dirs": ["latest_model_test", "data/chunks_test", "data/original_pdfs_test"], "deleted_metadata_count": 8}', 'adc21ad8-73d7-4748-b7bd-5c3337d5e219');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (87, '2025-11-18 19:34:11.623183', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "ANYJET User Guide V1.1.pdf", "machine_model": "Duraflex"}', '2a441144-2554-49bc-bf56-9f19ac2e1ea8');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (88, '2025-11-18 19:34:11.668996', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "ANYJET User Guide V1.1.pdf", "machine_model": "Duraflex", "metadata_id": "202c3a8d-ecd1-441f-845a-371089c6eff5", "status": "PENDING_INGESTION"}', '2a441144-2554-49bc-bf56-9f19ac2e1ea8');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (89, '2025-11-18 19:38:09.196915', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '23da077d-0270-4e75-8129-7a99cfdc6c32');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (90, '2025-11-18 19:43:28.470318', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '4efe9a01-8c4e-47ab-ba35-f605ab6dd03a');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (91, '2025-11-18 19:43:52.670229', 'info', 'document_deletion_started', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"metadata_id": "202c3a8d-ecd1-441f-845a-371089c6eff5", "filename": "ANYJET User Guide V1.1.pdf"}', '14270d0d-c399-4c7c-a360-b08330428187');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (92, '2025-11-18 19:44:19.811990', 'info', 'manual_upload_start', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "2800 Series Mini Laser Pro User Manual Draft.pdf", "machine_model": "2800 Series Mini Laser Pro"}', '1e6e8790-0230-4fc8-955e-3f2b61922baf');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (93, '2025-11-18 19:44:19.846493', 'info', 'document_metadata_created', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "2800 Series Mini Laser Pro User Manual Draft.pdf", "machine_model": "2800 Series Mini Laser Pro", "metadata_id": "186f5200-bd7d-4b4a-83ef-7e2776e4ea4d", "status": "PENDING_INGESTION"}', '1e6e8790-0230-4fc8-955e-3f2b61922baf');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (94, '2025-11-18 20:15:55.024537', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '3e362b7a-4409-478d-854e-ae84b1cafe6a');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (95, '2025-11-18 20:30:51.007099', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'd382d80b-c29b-4af7-bd0f-ce1e97fe5ab5');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (96, '2025-11-18 20:36:24.810918', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'd7e16501-4024-412e-9506-316f925dcdc5');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (97, '2025-11-18 20:36:48.003401', 'info', 'document_metadata_updated', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "Arrow Any-002 Manual.pdf", "updates": {"requires_admin_review": false, "machine_model": ["ANYTRON Any-002"]}}', '4b518e6a-75fb-48cd-85e6-646cc7c44846');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (98, '2025-11-18 20:36:59.222693', 'info', 'document_metadata_updated', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "ANYJET User Guide V1.1.pdf", "updates": {"requires_admin_review": false, "machine_model": ["Anytron AnyJet"]}}', 'd58304ac-727d-425c-8e59-94f35089f8d6');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (99, '2025-11-18 20:37:13.513303', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "print_quality_artefacts_reference_guide.pdf", "is_active": false, "status": "disabled"}', '9628b2a9-1a90-4e48-abc0-5f89d5ba577b');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (100, '2025-11-18 20:37:16.291520', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "print_quality_artefacts_reference_guide.pdf", "is_active": true, "status": "enabled"}', '6abcc020-04c9-4751-81d6-29a1d1a8aa83');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (101, '2025-11-18 20:41:41.690462', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '9155e194-e8e5-4fc0-885e-a787d3de27ff');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (102, '2025-11-18 20:46:19.962559', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'e410271a-0111-4041-8b18-9492832644f0');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (103, '2025-11-18 20:50:10.361845', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "print_quality_artefacts_reference_guide.pdf", "is_active": false, "status": "disabled"}', '2acaf032-b88a-4942-a8c5-2fe8bf45735b');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (104, '2025-11-18 20:50:42.486742', 'info', 'document_toggled', 'ethan@gmail.com', 'ADMIN', '172.18.0.1', '{"filename": "print_quality_artefacts_reference_guide.pdf", "is_active": true, "status": "enabled"}', 'fd50841d-ba68-4974-952f-b90f67fc9377');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (105, '2025-11-18 21:04:10.019757', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'bf79e575-9ac7-4bcc-bebb-b43421acba95');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (106, '2025-11-18 21:09:27.749416', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'f692ab48-d5ca-49ed-a335-9f7efffc4782');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (107, '2025-11-18 21:16:45.849583', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '039d5744-0c31-408a-82a0-87072c47bf36');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (108, '2025-11-18 21:26:35.924583', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', '5d4e0043-730c-4646-8b18-e6d0cf2c93bd');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (109, '2025-11-18 21:34:03.534729', 'warning', 'user_login_failed', 'ethan@gmail.com', NULL, '172.18.0.3', '{"reason": "invalid_credentials"}', '9d18f16c-32d1-4753-af74-7967085ba246');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (110, '2025-11-18 21:34:10.048067', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'bcbeb847-e56b-43c7-b40a-a41241fd228c');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (111, '2025-11-18 21:40:34.007362', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'b722292c-60e1-4295-9201-bc9afc374340');
INSERT INTO audit_logs (id, timestamp, level, event, user_id, role, ip_address, metadata, request_id) VALUES (112, '2025-11-18 21:47:15.740072', 'info', 'user_login', 'ethan@gmail.com', 'ADMIN', '172.18.0.3', '{"user_id": "3"}', 'ca7ce0d0-ce7f-482d-9386-08de588e1151');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (2, 6, 'Test', 'Definition: Top of Form: Electrical input signal synchronizing printing start with top of page location

Based on the provided documents, "test" refers to various diagnostic and verification procedures performed on industrial equipment and systems. Here are the key technical definitions and applications:

## Primary Test Categories

**Installation Testing**: Tests are performed after equipment installation to verify proper system functionality [2][3]. These tests ensure that newly installed equipment operates according to specifications before being put into service.

**Insertion Testing**: A specific diagnostic procedure where printhead insertion tests are performed at system initialization and subsequently during printing operations [6]. When this test fails, an "INSERTION_TEST_FAILED" error message is generated in system logs.

**Manual Testing Interface**: An interactive testing environment that allows operators to perform controlled equipment operations [9]. For example, in cutting equipment, operators can access a screen-based manual testing interface to activate functions like the knife press button for blade adjustment procedures.

## Test Procedures and Implementation

**Fault Detection Testing**: Systems continuously monitor for problems such as out-of-ink conditions or missing hardware components [4]. When issues are detected, the system enters a FAULT state, requiring diagnostic testing to identify and resolve the underlying causes.

**Status Verification Testing**: Equipment status can be tested through command-line interfaces using specific input strings [7]. For example, checking valve status returns responses like "UltronBinaryActuatorStatus.IDLE: 1", with expected responses dependent on driver state and physical position.

**Operational Parameter Testing**: Regular testing involves monitoring critical parameters including:
- Liquid inlet pressure (must not exceed maximum specified values)
- Temperature ranges (within minimum and maximum thresholds)
- Pressure drop measurements (significant increases indicate clogging)
- Vacuum line integrity (continuous liquid leaking indicates end-of-life conditions) [10]

## Testing Applications

Tests serve multiple purposes across different equipment types, from degassing modules that remove dissolved gas from liquids [1] to cutting systems that require precise blade positioning and depth adjustment [9]. Sample preparation testing involves cutting materials into specific configurations while preserving critical identification codes [8].

In summary, testing in industrial contexts encompasses systematic verification procedures that ensure equipment functionality, diagnose faults, monitor operational parameters, and validate system performance across installation, operation, and maintenance phases.', 45896, '{"sessionId": "session_1762973623681", "intentType": "definition", "intentConfidence": 0.3, "confidence": 0.5399727319553872, "sources": ["Unknown", "anyCUTII User Manual_ver1.0_ENG (1).pdf", "anyCUTIII User Manual_ver1.1_ENG_Ana (2).pdf", "DuraFlex Software Databook and Design Guide R3.1.1_V1.00_09Jun2020.pdf", "EZCut 350R Manual (1).docx.pdf"]}', '2025-11-12 18:54:29.594628', NULL, NULL, NULL, NULL, NULL, NULL);
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (3, 6, 'Duraflex System Requirements?', 'Definition: DuraFlex Ink: Memjet aqueous pigment ink used with DuraFlex Print Engine

# DuraFlex System Requirements

Based on the provided documentation, the DuraFlex system has specific hardware and software requirements that vary depending on performance needs and application requirements.

## Standard Configuration Requirements

The DuraFlex system requires the following minimum specifications [2]:

**Processor:**
- 3.40 GHz 16-Core Processor (32 threads)
- 4.90 GHz Max Turbo Frequency
- Closed Loop Water Cooler [2]

**Memory and Storage:**
- 32GB DDR4-3600 MHz Memory
- 1 x Samsung 500GB 970 EVO Plus NVMe SSD
- 1 x 960GB Enterprise Solid State Drive [2]

**Network Components:**
- 1 x Intel 2 Port – Ethernet Converged Network Adapter X710-DA2 (X710DA2BLK)
- 1 x Intel Ethernet SFP+SR Optic (E10GSFPSR) [2]

**Graphics and Operating System:**
- GeForce GT710 Graphic Card
- Microsoft Windows 11 Pro 64 bit [2]

## Alternative Configuration

An alternative configuration is also documented with the following specifications [3][4]:

**Processor:**
- 2.10 GHz Base/4.90 Max Turbo
- 12 Core/20 Thread processor [3]

**Memory and Storage:**
- 32GB DDR5 3200MHz Memory
- One (1) 500GB Samsung 980 PRO NVMe [4]

**Network and Graphics:**
- One (1) Intel Gigabit CT Desktop Adapter
- One (1) Intel 2 Port – 10Gbase-T Ethernet Converged Network Adapter
- One (1) GeForce GT 730 Graphic Card
- One (1) Microsoft Windows 11 Pro 64 bit [4]

## High Performance Configuration

For enhanced performance requirements, particularly for suboptimal input files, 2-wide or duplex applications, or POD systems requiring real-time page requests in any order, a higher performance configuration is recommended [4]:

**Processor:**
- AMD Ryzen 9 5950X CPU [4]

## System Integration

The DuraFlex system integrates with various mechanical components and control systems, including compatibility with Kirrawee/Ultron mechanical control systems [5]. The system also supports command line interface operations for component control including circulation pumps, valves, lifter motors, and maintenance servo motors [6][8][9].

## Summary

The DuraFlex system requirements encompass multiple configuration options ranging from standard to high-performance setups, all requiring Windows 11 Pro 64-bit operating system, substantial memory (32GB), NVMe SSD storage, and specialized network adapters. The specific configuration choice depends on the intended application requirements and performance expectations [2][4].', 44370, '{"sessionId": "session_1763061775007", "intentType": "lookup", "intentConfidence": 0.9, "confidence": 0.9967857142857144, "sources": ["DuraFlex TB0002 Printhead Identification and System Compatibility 20Nov2020.pdf", "DuraFlex System Requirements _ Xitron Support.pdf", "DuraFlex_Command Line Interface-Supplimental_V1.0_29April2025.pdf"]}', '2025-11-13 19:23:39.387951', 'DuraFlex', 3231, 673, 3904, 0.019788, '["DuraFlex TB0002 Printhead Identification and System Compatibility 20Nov2020.pdf", "DuraFlex System Requirements _ Xitron Support.pdf", "DuraFlex_Command Line Interface-Supplimental_V1.0_29April2025.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (4, 6, 'Network not connected', 'Definition: GbE: Gigabit Ethernet - high-speed network connection

# Network Connection Troubleshooting

When encountering "Network not connected" errors, the issue typically involves communication failures between the PC and machine, requiring systematic verification of physical connections, network configurations, and software settings.

## Primary Troubleshooting Steps

### 1. Physical Connection Verification
First, verify all physical network connections. For EZCut systems, **check the network connection line** [5]. If using serial communication, **check the connection wire and re-insert the port** [1]. Ensure that **the PC port and software port are functioning correctly** [1].

### 2. Network IP Configuration
Network connectivity issues often stem from incorrect IP settings:
- **Set PC network IP as per section 4-2-3** of the respective manual [5][9]
- **Set machine network IP as per section 4-2-4** of the respective manual [5][9]
- Verify that **the machine status is displayed on the main interface** [4]

### 3. Software and Firewall Issues
Software conflicts can cause network communication failures. When **the computer screen is connected, but the software says it''s not online**, the solution is to **disable virus detection software and re-install cutting software** [3]. This addresses **network failure as a result of anti-virus software firewall** [3].

## Advanced Network Issues

### Hardware-Level Network Errors
For systems displaying **Error 063: Network Communication Error**, this indicates an **H/W I/F problem between CU and NIC** [2]. The troubleshooting sequence is:
1. **Turn the power OFF/ON** [2]
2. If the error occurs repeatedly, **replace the network card (C9600)** [2]
3. As a final step, **replace the CU board** [2]

### High-Speed Network Configuration
For DuraCore systems with 10G Ethernet connections, **if the 10G Ethernet network lights turn off after the print engine boots up, it is likely due to the Ethernet card in the RIP PC not being compatible with 5G speeds** [7]. To resolve this:
1. **Add a file named network.conf to the print engine directory /mnt/durabolt_config/system/** [7]
2. **Include the line: FORCE_10G_RIP_DATA_SPEED=1** [7]
3. **Reboot the print engine** [7]

## System-Specific Considerations

### Duplex System Network Management
For duplex print systems, **as both engine stages are connected to the same network switch, the IP addresses will clash until the stage 2 unit is reconfigured** [8]. The recommended approach involves:
1. **Disconnect the 1G network connection to the Stage 2 PM1 Mechatronic Controller PCBA and Datapath PCBA** [8]
2. Test Stage 1 connectivity first
3. **Wait about half a minute for the system to start up** [8][10]
4. Systematically test each stage individually

### Power and Grounding Verification
If network issues persist, verify the power infrastructure:
- **Check that your machine and power supply are grounding** [1]
- **Check voltage of power supply is stable** [1]
- **Check the main board and panel board for damage** [1]

## Conclusion

Network connectivity issues require systematic troubleshooting starting with physical connections, followed by IP configuration verification, and software conflict resolution. For persistent issues, hardware replacement of network components or control boards may be necessary. **If you can''t solve this problem, please contact Arrow Systems Inc** [1] or the appropriate technical support for your specific system.', 65043, '{"sessionId": "session_1763152844326", "intentType": "troubleshooting", "intentConfidence": 0.95, "confidence": 0.7675791179251672, "sources": ["EZCut 330 User Manual Ver.202106 (1).pdf", "Arrow Any-002 Manual.pdf", "EZCut 330 Troubleshooting.pdf", "EZCut 350R Manual (1).docx.pdf", "DuraCore Operating and Troubleshooting 01May2025.pdf", "DuraCore 1x1 Print Engine Installation Guide_v2.0_18Aug2025.pdf"]}', '2025-11-14 20:41:49.383759', NULL, 2316, 830, 3146, 0.019398, '["EZCut 330 User Manual Ver.202106 (1).pdf", "Arrow Any-002 Manual.pdf", "EZCut 330 Troubleshooting.pdf", "EZCut 350R Manual (1).docx.pdf", "DuraCore Operating and Troubleshooting 01May2025.pdf", "DuraCore 1x1 Print Engine Installation Guide_v2.0_18Aug2025.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (5, 6, '350R Network not connected', 'Definition: Top of Form: Electrical input signal synchronizing printing start with top of page location

## EZCut 350R Network Connection Troubleshooting

The "Network not connected" issue on the EZCut 350R indicates a **failure connection between PC and host network** [5]. This problem requires systematic verification of both physical connections and network configuration settings.

## Step-by-Step Troubleshooting Procedure

### 1. Physical Connection Verification
First, **check the network connection line** [2] to ensure all cables are properly connected between the PC and the EZCut 350R machine.

### 2. PC Network IP Configuration
**Set PC network IP as per section 4-2-3** [2][3] of the manual. This step is critical for establishing proper communication between your computer and the machine.

### 3. Machine Network IP Configuration
**Set machine network IP as per section 4-2-4** [2][3] of the manual. Both the PC and machine must be configured with compatible network settings to communicate effectively.

### 4. Machine Status Verification
Ensure that **the machine status is displayed on the main interface** [1]. If the status is not visible, this indicates the network connection has not been established successfully.

## Additional Considerations

The troubleshooting documentation consistently emphasizes three core areas across multiple references [1][2][3]:
- Physical network connection integrity
- PC network IP configuration (section 4-2-3)
- Machine network IP configuration (section 4-2-4)

## Resolution Summary

To resolve the EZCut 350R network connection issue, systematically verify the physical network cable connection, then configure both PC and machine network IP settings according to the specified manual sections. The successful establishment of network communication should result in the machine status appearing on the main interface, confirming proper connectivity between the PC and the EZCut 350R system.', 72964, '{"sessionId": "session_1763152924370", "intentType": "troubleshooting", "intentConfidence": 0.95, "confidence": 0.9305698079106524, "sources": ["EZCut 350R Manual (1).docx.pdf", "Arrow Any-002 Manual.pdf", "EZCut 330 User Manual Ver.202106 (1).pdf"]}', '2025-11-14 20:43:17.344176', NULL, 2271, 408, 2679, 0.012933, '["EZCut 350R Manual (1).docx.pdf", "Arrow Any-002 Manual.pdf", "EZCut 330 User Manual Ver.202106 (1).pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (6, 6, 'network connection error ', 'Definition: GbE: Gigabit Ethernet - high-speed network connection

# Network Connection Error Troubleshooting

Network connection errors can occur across multiple system components and require systematic troubleshooting to identify and resolve the root cause.

## Primary Network Error Types

**Error 063: Network Communication Error** - This indicates a hardware interface problem between the Control Unit (CU) and Network Interface Card (NIC). When this error occurs repeatedly, follow these steps [1]:

1. Turn the power OFF/ON
2. Replace the network card (C9600)
3. Replace the CU board if the issue persists

**Communication Errors Between System Boards** - Interface errors between Processing Unit (PU) and Control Unit (CU) can manifest as network-related issues. For these communication errors [7]:

1. Verify the CU board is correctly installed
2. If incorrectly installed, reinstall the CU board properly
3. If correctly installed but error persists, replace the CU board
4. If CU board replacement doesn''t resolve the issue, replace the PU board

## Software-Related Network Issues

**Driver and Interface Errors** include [1]:
- **Error 052**: DMA image process driver data error - Reinstall the driver, then power cycle if error repeats
- **Error 060**: Parallel Interface Driver Error - Power cycle the system
- **Error 062**: USB Driver Error - Power cycle the system

**Network Communication Failures** can result from software conflicts. Anti-virus software firewalls may block network connections, causing the system to appear offline despite physical connectivity [10]. In such cases:

1. Disable virus detection software temporarily
2. Reinstall the cutting software
3. Reconfigure firewall settings to allow system communication

## Hardware Connection Verification

**Physical Connection Issues** require systematic checking [3]:
- Verify all connection wires are properly seated
- Re-insert communication ports
- Confirm PC port and software port compatibility
- Check for signal interference from other equipment

**Power and Grounding Issues** can cause communication errors [3]:
- Ensure machine and power supply are properly grounded
- Verify power supply voltage stability
- Check for damage to main board and panel board components

## Advanced Troubleshooting

**Engine Communication Errors** may present as network issues when the system reports "RESULT_COMMS_ERROR: Thrift exception: connect() failed: Connection refused" [9]. This indicates a fundamental communication failure between system components.

**RFID and Interface Errors** can also manifest as network problems [2]:
- **Error 230**: RFID Reader connection issues - Check RFID R/W board connections
- **Error 231**: TAG Interface or RFID Reader I/F connection errors - Verify antenna cable connections and replace boards as necessary

## Resolution Summary

Network connection errors require a methodical approach starting with power cycling, followed by connection verification, driver reinstallation, and component replacement if necessary. Most network communication issues can be resolved through proper hardware installation verification and systematic component replacement, beginning with network cards and progressing to control boards when simpler solutions fail to resolve the problem.', 68065, '{"sessionId": "session_1763153372891", "intentType": "troubleshooting", "intentConfidence": 0.95, "confidence": 0.8452931836811679, "sources": ["Arrow Any-002 Manual.pdf", "EZCut 330 User Manual Ver.202106 (1).pdf", "DuraFlex Troubleshooting Guide V4.05_30May2022.pdf", "EZCut 330 Troubleshooting.pdf"]}', '2025-11-14 20:50:40.973305', NULL, 3484, 683, 4167, 0.020697, '["Arrow Any-002 Manual.pdf", "EZCut 330 User Manual Ver.202106 (1).pdf", "DuraFlex Troubleshooting Guide V4.05_30May2022.pdf", "EZCut 330 Troubleshooting.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (7, 8, 'test', 'Definition: Top of Form: Electrical input signal synchronizing printing start with top of page location

I cannot provide a comprehensive technical answer to your query because "test" is not a specific technical question that can be addressed using the provided documents.

The provided documents contain technical information about:

- Print quality artefacts and troubleshooting procedures [1][8]
- EZCut 350R cutting machine operation and maintenance [2][3][5][6][7][9][10]
- Equipment warranty terms and conditions [4]

To generate a proper technical response, please provide a specific question related to:

1. **Print Quality Issues**: Such as horizontal line variations, color mixing problems, or other printing artefacts
2. **EZCut 350R Operations**: Including blade replacement, software interface usage, cutting procedures, or troubleshooting
3. **Equipment Warranty**: Regarding warranty terms, compatibility testing, or remedy procedures

**Recommendation**: Please reformulate your query with a specific technical question so I can provide accurate, document-based information with proper citations and technical details.', 51849, '{"sessionId": "session_1763154032053", "intentType": "definition", "intentConfidence": 0.3, "confidence": 0.5044169333105171, "sources": ["print_quality_artefacts_reference_guide.pdf", "EZCut 350R Manual (1).docx.pdf", "OperationManualEFSmallSeries_en.pdf"]}', '2025-11-14 21:01:23.920013', NULL, 2136, 218, 2354, 0.009677999999999999, '["print_quality_artefacts_reference_guide.pdf", "EZCut 350R Manual (1).docx.pdf", "OperationManualEFSmallSeries_en.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (8, 8, 'How do i set up the Lamination?', 'Definition: Top of Form: Electrical input signal synchronizing printing start with top of page location

# Lamination Setup Procedure for EZCut 350R

To set up the lamination on the EZCut 350R digital die cutter, follow this systematic procedure:

## Step-by-Step Setup Process

**1. Load the Laminating Film Roll**
Load a laminating film roll on the laminating expand shaft and adjust the appropriate position to be parallel to the covered area [1]. The width of laminating film must be less than the media width [3].

**2. Adjust Tension Settings**
Tension the lamination shaft after positioning the film roll [1]. The laminating expand shaft operates as a passive system using main feeding power without a dedicated driving motor [1]. You can adjust the tension through the software menu in parameter settings, with a suggested tension range of 15-20 [1]. This tension adjustment controls friction and overshoot capacity [1].

**3. Install Cold Lamination Film**
After installing the cold lamination film, stick the adhesive side onto the material [1]. Ensure proper alignment and contact between the film and substrate material.

**4. Final Positioning and Release**
Click the forward button on the control panel to flatten the film and release the lamination roller [1]. This step ensures proper film application and prepares the system for operation.

## Technical Specifications

The EZCut 350R supports a maximum laminating roll diameter of 300mm [6] and is equipped with high-end laminating film compatibility through its linear system design [4]. The lamination tension can be monitored and adjusted via the control interface, which includes dedicated lamination tension controls [10].

## Summary

The lamination setup involves loading the film roll, adjusting software-controlled tension parameters (15-20 recommended), applying the adhesive side to the material, and using the control panel''s forward button to complete the setup. The passive shaft system relies on main feeding power and requires proper tension calibration for optimal performance.', 86652, '{"sessionId": "session_1763154095025", "intentType": "reasoning", "intentConfidence": 0.92, "confidence": 0.5671339400212223, "sources": ["EZCut 350R Manual (1).docx.pdf"]}', '2025-11-14 21:03:01.692987', NULL, 2239, 427, 2666, 0.013122000000000002, '["EZCut 350R Manual (1).docx.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (9, 8, 'What are the duraflex system requirements?', 'Definition: Mechanical Controller PCA: PCA board controlling motors and sensors on all DuraFlex modules

According to EZCut 350R Manual (1).docx.pdf [1]:
Note: The camera parameters are within the system have been debugged in the factory and no need to be changed....

According to EZCut 350R Manual (1).docx.pdf [2]:
Wipe all the slide rails of the motion system with alcohol and cotton cloth. The rollers

must

be

kept

clean,

because

dust

will

block

the

rim

and

cause

the

wear

of

the

rollers,

making

the

movement

system

uneven.

After

the

slide

rail

is

cleaned,

use

a

clean

cotton

swab

moistened

with

alcohol

against

the

roller,

and

move

the

motion

system

with

the

other

hand

to

rotate

the

roller

while

cleaning

the

roller

and

the

slide

rail.

7-3

Trouble

...

According to EZCut 350R Manual (1).docx.pdf [3]:
Chapter

Machine

maintenance

7-1

Maintenance

Check the media feeding system parts every day before use. If there is some dust or

debris,

please

clean

shaft,

roller

and

board

with

alcohol

and

gasoline,

You

must

do

the

basic

clean,

the

clean

rate

depends

on

your

use

and

the

media.

Dust

will

be

blocking

the

die-cutting

system

moving.

Caused

the

moving

incorrect,

even

there

is

risk

and

effect

the

life-span

of

equipment.

7-2

General

cleaning

an...', 49421, '{"sessionId": "session_1763392270267", "intentType": "lookup", "intentConfidence": 0.95, "confidence": 0.647246377020737, "sources": ["EZCut 350R Manual (1).docx.pdf", "print_quality_artefacts_reference_guide.pdf"]}', '2025-11-17 15:11:59.714032', 'DuraFlex', NULL, NULL, NULL, NULL, '["EZCut 350R Manual (1).docx.pdf", "print_quality_artefacts_reference_guide.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (10, 8, 'What are the duraflex system requirements?', 'Definition: Mechanical Controller PCA: PCA board controlling motors and sensors on all DuraFlex modules

Based on the provided documents, I cannot find specific information about "DuraFlex system requirements" for either the DuraBolt or EZCut 350R machines that you own.

The documents contain various technical specifications and requirements for your machines, but none specifically reference a "DuraFlex" system. The available documentation includes:

**For the EZCut 350R:**
- Space requirements (floor size) specifications [3]
- Environmental conditions including humidity requirements of 30~90% [3]
- Machine handling requirements noting wheels at the bottom for movement and forklift compatibility [3]
- Camera system parameters that are factory-debugged and require no changes [1]
- Motion system maintenance requirements including slide rail cleaning procedures [2]

**General System Information:**
The documents reference various other systems and components, but do not contain specific "DuraFlex" system requirements or specifications.

**Conclusion:**
The term "DuraFlex" does not appear in the provided technical documentation for your DuraBolt and EZCut 350R machines. To obtain accurate DuraFlex system requirements, you may need to consult additional technical documentation specific to that system, or contact technical support for clarification on whether this system is applicable to your particular machine models.

If you are referring to a different system or component by a similar name, please provide additional context so I can locate the relevant specifications in your machine documentation.', 67347, '{"sessionId": "session_1763392560429", "intentType": "lookup", "intentConfidence": 0.95, "confidence": 0.7328941830485478, "sources": ["EZCut 350R Manual (1).docx.pdf", "Unknown", "print_quality_artefacts_reference_guide.pdf"]}', '2025-11-17 15:17:07.789125', 'DuraFlex', 2162, 311, 2473, 0.011151000000000001, '["EZCut 350R Manual (1).docx.pdf", "Unknown", "print_quality_artefacts_reference_guide.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (11, 8, 'How do I Adjust the blade depth on the 350R?', 'Definition: Mechanical Controller PCA: PCA board controlling motors and sensors on all DuraFlex modules

## Blade Depth Adjustment on the EZCut 350R

To adjust the blade depth on the EZCut 350R, you need to set the blade length based on your media thickness and fine-tune it through operational testing.

### Step-by-Step Procedure

**Step 1: Determine Required Blade Length**
First, measure your media thickness "t" [1]. The blade length should be equal to or slightly longer than the media thickness, but must not exceed the total thickness of both the top and bottom paper layers [1].

**Step 2: Initial Blade Length Setting**
Set the initial blade length based on your media thickness measurement. Ensure the blade penetrates through the top material but does not cut completely through the bottom backing paper [1].

**Step 3: Fine-Tune Through Testing**
If you cannot accurately determine the top paper thickness, use the directional control keys on the machine to incrementally adjust the blade length [1]. Increase the blade depth bit by bit while performing test cuts until the blade leaves only light traces on the bottom paper after cutting [1].

**Step 4: Verify Cutting Quality**
Perform trial cuts using the "trial cut" function on the screen to evaluate the cutting effect and position [4]. The goal is to achieve complete cutting of the intended material while preserving the integrity of the backing material.

### Important Considerations

- **Precision Control**: The EZCut 350R offers 0.1mm slitting precision [6], allowing for very fine blade depth adjustments
- **Material Protection**: Proper blade depth prevents damage to the bottom paper while ensuring complete cuts through the top material [1]
- **Incremental Adjustment**: Use small incremental changes when fine-tuning to avoid over-cutting [1]

### Summary

Blade depth adjustment on the EZCut 350R is achieved by setting the blade length equal to or slightly greater than your media thickness, then fine-tuning through incremental adjustments using the directional keys until optimal cutting is achieved with minimal marking on the backing material [1].', 197478, '{"sessionId": "session_1763392641331", "intentType": "reasoning", "intentConfidence": 0.95, "confidence": 0.8766666666666667, "sources": ["EZCut 350R Manual (1).docx.pdf"]}', '2025-11-17 15:20:38.825252', NULL, 2231, 459, 2690, 0.013578, '["EZCut 350R Manual (1).docx.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (12, 8, 'How many mm does each rotation of the knob cause?', 'Definition: Lift Mechanism: Moves printhead cradle to home cap/wipe and print positions

Based on the provided documentation, the blade depth adjustment knob on the EZCut 350R causes the blade to move by **approximately 0.1mm per rotation**.

## Technical Details

The blade depth adjustment mechanism is located on the EZCut 350R''s manual testing interface. When adjusting blade depth, you access this function by clicking on the knife press button on the screen [3].

## Adjustment Procedure

The blade depth adjustment follows this process:

1. Access the manual testing interface on the screen
2. Click on the knife press button
3. Rotate the blade length adjustment knob to modify blade extension
4. Each complete rotation of the knob moves the blade approximately 0.1mm [3]

## Directional Control

The knob provides bidirectional blade adjustment:
- **Extending direction**: Rotates the knob to increase blade protrusion
- **Retracting direction**: Rotates the knob to decrease blade protrusion [3]

## Critical Safety Consideration

Proper blade depth adjustment is essential for equipment protection. If the blade protrudes longer than the thickness of the media being cut, it will damage both the blade tip and the holding bar [3]. The 0.1mm precision per rotation allows for fine-tuned adjustments to achieve optimal cutting depth for different media thicknesses.

## Summary

The EZCut 350R blade depth adjustment knob provides precise control with approximately 0.1mm of blade movement per complete rotation, enabling accurate blade positioning for various cutting applications while preventing equipment damage through over-extension.', 62123, '{"sessionId": "session_1763392975480", "intentType": "lookup", "intentConfidence": 0.95, "confidence": 0.7573078602478847, "sources": ["EZCut 350R Manual (1).docx.pdf", "DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "print_quality_artefacts_reference_guide.pdf"]}', '2025-11-17 15:23:57.617032', NULL, 2624, 349, 2973, 0.013107, '["EZCut 350R Manual (1).docx.pdf", "DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "print_quality_artefacts_reference_guide.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (13, 8, 'are you sure its not .5 mm', 'Definition: DPI: Dots Per Inch - linear measurement of resolution for scanning and printing

I need more context to provide an accurate answer about what specific measurement you''re questioning regarding the 0.5mm value. However, based on the provided documentation for your DuraBolt and EZCut 350R machines, I can clarify the key measurements that are specified:

## DuraBolt Calibration Measurements

For the DuraBolt system, the documented calibration specifications are **0.35mm**, not 0.5mm:

**Print Height Calibration:**
- The standard calibration gap is **0.35mm** using a feeler gauge [5]
- The procedure requires raising or lowering the calibration position "until the gap is a comfortable fit for the 0.35mm feeler gauge on BOTH SIDES but too tight for the 0.4mm gauge" [5]

**Physical Installation Setup:**
- During print engine installation, a **0.35mm shim** is placed on the parallel block for height adjustment [9]
- The setup requires that "a 0.4mm shim should not be able to fit between the nest calibration pads and the parallel blocks" [9]

## Media Encoder Accuracy Requirements

For media timing calibration, the DuraBolt system requires accuracy of **better than 0.1%**, not 0.5mm [8]. This is a percentage measurement for encoder calibration, not a dimensional measurement.

## EZCut 350R Specifications

The EZCut 350R documentation provided does not contain any 0.5mm specifications. It focuses on pixel scaling conversions and cutting position parameters [6][7].

**Conclusion:** Based on the technical documentation for your machines, the standard calibration measurement for the DuraBolt system is consistently specified as **0.35mm**, not 0.5mm. If you''re referring to a different measurement or specification, please provide additional context so I can give you a more targeted response.', 64898, '{"sessionId": "session_1763393126487", "intentType": "lookup", "intentConfidence": 0.85, "confidence": 0.8138741573490933, "sources": ["DuraBolt DuraCore Software Release Notes R3.3.0_V1.01_05Sep2025.pdf", "DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "EZCut 350R Manual (1).docx.pdf"]}', '2025-11-17 15:26:31.403129', NULL, 3500, 426, 3926, 0.016890000000000002, '["DuraBolt DuraCore Software Release Notes R3.3.0_V1.01_05Sep2025.pdf", "DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "EZCut 350R Manual (1).docx.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (14, 8, '2', 'Definition: Top of Form: Electrical input signal synchronizing printing start with top of page location

I cannot provide a comprehensive technical answer to your query "2" as this appears to be an incomplete or unclear request.

Based on the provided documents, I have access to technical information about your EZCut 350R machine, including:

- Power-up procedures: "Turn on the main power and wait for 10 seconds for the computer to turn on before turning on the diecutting" [2]
- Blade system specifications: "This cutting machine uses cutting blades installed in the holder for cutting. There are two different blade holders to fit the installed blade diameter" [3]

However, your query "2" does not specify what technical details or specifications you need regarding your EZCut 350R machine.

**To provide you with accurate technical assistance, please clarify:**
- Are you looking for information about a specific component numbered "2"?
- Do you need step-by-step procedures for a particular operation?
- Are you requesting troubleshooting guidance for an error code or issue?
- Do you need specifications for a particular system or part?

Once you provide a more specific query about your EZCut 350R machine, I can deliver comprehensive technical details using the available documentation.', 46110, '{"sessionId": "session_1763394144401", "intentType": "lookup", "intentConfidence": 0.3, "confidence": 0.47005705700976286, "sources": ["Unknown", "EZCut 350R Manual (1).docx.pdf"]}', '2025-11-17 15:43:10.525389', '2800 Series Mini Laser Pro', 2080, 263, 2343, 0.010185, '["Unknown", "EZCut 350R Manual (1).docx.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (15, 8, 'How to check and adjust the verticality on the 350R?', 'Definition: Lift Mechanism: Moves printhead cradle to home cap/wipe and print positions

I cannot provide information about checking and adjusting verticality on the 350R model, as this specific model is not referenced in the provided documentation for your DuraBolt machine.

The provided documents contain detailed installation and calibration procedures for the DuraBolt system, including:

**Available Calibration Procedures:**
- Print height calibration using 0.35mm feeler gauges and parallel blocks [8][10]
- Print engine angular alignment corrections with specific torque requirements (20-25 Nm) [6]
- Printhead angular misalignment corrections for Tandem systems [6]
- Fine alignment chart printing and scanning procedures [2]

**Height Adjustment Process Available:**
The DuraBolt documentation shows procedures for adjusting print engine height using adjustable feet, where a 0.35mm shim should fit snugly on both sides of the nest calibration pads, while a 0.4mm shim should not fit [10].

**Conclusion:**
The provided documentation does not contain specific procedures for checking and adjusting verticality on a 350R model. To obtain the correct verticality adjustment procedure for your specific equipment, you would need to consult the appropriate technical documentation for the 350R model or contact technical support for guidance on your particular machine configuration.', 97790, '{"sessionId": "session_1763394708764", "intentType": "reasoning", "intentConfidence": 0.92, "confidence": 0.7246664852907896, "sources": ["Unknown", "DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "print_quality_artefacts_reference_guide.pdf", "OperationManualEFSmallSeries_en.pdf"]}', '2025-11-17 15:53:26.570163', NULL, 2827, 284, 3111, 0.012741, '["Unknown", "DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "print_quality_artefacts_reference_guide.pdf", "OperationManualEFSmallSeries_en.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (16, 8, 'Tell me about the Logs Page on durabolt?', 'Definition: Top of Form: Electrical input signal synchronizing printing start with top of page location

Based on the provided documentation, the Logs Page on the DuraBolt system is accessed through the DuraBolt Management Interface (DMI) and is referred to as the "DMI Log Files screen."

## DMI Log Files Screen Overview

The DMI Log Files screen is part of the DuraBolt Management Interface (DMI), which serves as one of the primary configuration tools for the DuraBolt system [2]. This interface provides access to system log files for monitoring, troubleshooting, and maintenance purposes [3].

## Access Method

The Logs Page is accessed through the DMI, which is the main management interface for DuraBolt operations [2]. The DMI provides a comprehensive interface for system configuration and monitoring, with the Log Files screen being one of its key components for system diagnostics [3].

## Interface Context

The DMI Log Files screen is part of the broader DMI system that has been enhanced in recent software releases. The DMI interface has undergone redesign improvements, including a redesigned Control tab that provides easier access to common operator operations [5]. This suggests that the log files functionality is integrated into a user-friendly interface designed for operational efficiency.

## Technical Integration

The log files functionality is integrated with other DMI features, including:
- System configuration tools [2]
- Snapshot backup and restore capabilities [1][10]
- Print engine management functions [5]
- System monitoring and control operations [5]

## Documentation Reference

The DMI Log Files screen is specifically illustrated in the DuraBolt Installation Guide as Figure 122, indicating its importance as a standard operational tool for system administrators and technicians [3].

## Summary

The Logs Page on the DuraBolt system is implemented as the "DMI Log Files screen" within the DuraBolt Management Interface. This screen provides access to system log files for monitoring and troubleshooting purposes, and is integrated into the comprehensive DMI interface that serves as the primary configuration and management tool for DuraBolt operations [2][3].', 66169, '{"sessionId": "session_1763400196043", "intentType": "definition", "intentConfidence": 0.9, "confidence": 1.0, "sources": ["DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "DuraBolt DuraCore Software Release Notes R3.3.0_V1.01_05Sep2025.pdf"]}', '2025-11-17 17:24:22.226463', 'DuraBolt', 3862, 439, 4301, 0.018171, '["DuraBolt Installation Guide_v6.0_18Aug2025.pdf", "DuraBolt DuraCore Software Release Notes R3.3.0_V1.01_05Sep2025.pdf"]');
INSERT INTO query_history (id, user_id, query_text, answer_text, response_time_ms, metadata, created_at, machine_name, token_input, token_output, token_total, cost_usd, sources_json) VALUES (17, 8, 'How do i make the cut file?', 'Definition: RIP: Raster Image Processor - converts printer language to printhead image

## Creating Cut Files for the EZCut 350R

To create cut files for your EZCut 350R, you need to prepare files in the correct format and configure them properly for the die-cutting process.

### File Format Requirements

The EZCut 350R accepts files in **PLT and DXF formats** [5]. For optimal results, you must export your die-cutting file to **PLT format with specific settings**:
- Plotter unit: **1016**
- Curve resolution: **0** [2]

### Essential File Preparation Steps

**Step 1: Mark Configuration**
Regardless of your tracking method (single mark, double mark, or camera tracking), both your printed file and die-cutting file must include **double black marks positioned on the right side** [2].

**Step 2: Content Modifications**
- **Remove numbers**: Delete any numbers from the original printed file when creating the die-cutting file [2]
- **QR Code handling**: If using QR code functionality, replace the QR code in the die-cutting file with a black mark [2]

### Software Interface Configuration

Once your file is prepared, use the EZCut 350R software interface to configure cutting parameters:

**Tool Selection and Setup:**
- Choose knives based on your file''s graphic arrangement [1]
- Configure offset compensation for the tool axis [1]
- Enable "Maintain source file order" to cut according to graphic arrangement sequence [1]

**Automatic Calculations:**
- Use "Automatic calculation of tool drop point" to set drop points on X or Y axis according to graphics [1]
- Enable "Automatic calculation of tool spacing" for automatic tool head assignment and distance calculation between tools [1]

**File Management:**
- Utilize "Auto job change" function to recognize numbers or QR codes for automatic order switching [1]
- Enable "File transfer" to automatically delete completed work files and transfer them to the completed list [1]

### File Operations

The software provides standard file operations through the File menu for opening, saving, and exporting files [2]. You can also access optimization features for drop point setting and file optimization, plus output functions for regular and split output [2].

### Summary

Creating cut files for the EZCut 350R requires exporting to PLT format with specific settings (plotter unit 1016, curve resolution 0), ensuring proper black mark placement, removing numbers from die-cutting files, and configuring the software interface for optimal tool selection and automatic calculations. The system supports both manual and automatic file management features to streamline your cutting workflow.', 66242, '{"sessionId": "session_1763401315398", "intentType": "reasoning", "intentConfidence": 0.9, "confidence": 0.7526328632265927, "sources": ["EZCut 350R Manual (1).docx.pdf"]}', '2025-11-17 17:43:01.652392', NULL, 2603, 578, 3181, 0.016479, '["EZCut 350R Manual (1).docx.pdf"]');
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (1, 6, 2, True, 0.5399727319553872, 'definition', '2025-11-12 18:54:33.481930', NULL);
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (2, 6, 2, False, 0.5399727319553872, 'definition', '2025-11-12 18:54:38.367069', NULL);
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (3, 6, 3, True, 0.9967857142857144, 'lookup', '2025-11-13 19:26:52.424778', NULL);
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (4, 6, 3, False, 0.9967857142857144, 'lookup', '2025-11-13 19:27:01.905742', NULL);
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (5, 6, 3, True, 0.9967857142857144, 'lookup', '2025-11-13 19:27:07.253668', NULL);
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (6, 6, 10, True, 0.7328941830485478, 'lookup', '2025-11-17 15:17:27.431935', NULL);
INSERT INTO feedback (id, user_id, query_history_id, is_helpful, confidence, intent_type, created_at, updated_at) VALUES (7, 6, 10, False, 0.7328941830485478, 'lookup', '2025-11-17 15:17:30.334936', NULL);
-- Fix sequences
SELECT setval('users_id_seq', (SELECT COALESCE(MAX(id), 1) FROM users), true);
SELECT setval('query_history_id_seq', (SELECT COALESCE(MAX(id), 1) FROM query_history), true);
SELECT setval('saved_responses_id_seq', (SELECT COALESCE(MAX(id), 1) FROM saved_responses), true);
SELECT setval('feedback_id_seq', (SELECT COALESCE(MAX(id), 1) FROM feedback), true);
SELECT setval('audit_logs_id_seq', (SELECT COALESCE(MAX(id), 1) FROM audit_logs), true);
SELECT setval('machine_models_id_seq', (SELECT COALESCE(MAX(id), 1) FROM machine_models), true);
COMMIT;