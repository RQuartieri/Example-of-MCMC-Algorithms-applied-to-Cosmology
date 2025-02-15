import subprocess

subprocess.run(["python3", "FinalProject/Mockdata.py"])

print("Metropolis–Hastings algorithm:")
subprocess.run(["python3", "FinalProject/FinalProject_MH.py"])

print("emcee algorithm:")
subprocess.run(["python3", "FinalProject/FinalProject_emcee.py"])

# se quiser testar somente um dos algoritimos comente a linha do que deseja excluir aqui e rode o programa main