import os

# Caminho base do projeto
base_path = r"C:\Users\fferr\Desktop\ALURA\formacao_nocode\cp_refrigeration\embraco_predictive_platform"

def add_init_files(base_path):
    for root, dirs, files in os.walk(base_path):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(f"# Pacote Python: {os.path.basename(root)}\n")
            print("Criado:", init_file)

if __name__ == "__main__":
    add_init_files(base_path)