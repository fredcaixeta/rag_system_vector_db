from pymilvus import connections, utility

# Conecte-se ao Milvus
connections.connect("default", host="localhost", port="19530")

# Verifique se o servidor está respondendo
print(utility.get_server_version())  # Deve retornar a versão (ex: "2.3.3")