# scripts/collection_analyzer.py
import os
from dotenv import load_dotenv
from milvus_db import ZillizClient
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Carregar variáveis de ambiente
load_dotenv()

class CollectionAnalyzer:
    def __init__(self, collection_name="dr_voss_v2"):
        self.client = ZillizClient(
            api_key=os.getenv("ZILLIZ_API_KEY"),
            cluster_id=os.getenv("ZILLIZ_CLUSTER_ID"),
            region=os.getenv("ZILLIZ_REGION", "gcp-us-west1")
        )
        self.collection_name = collection_name
    
    def get_collection_stats(self):
        """Obtém estatísticas básicas da coleção"""
        stats = self.client.get_collection_stats(self.collection_name)
        print("\n📊 Estatísticas da Coleção:")
        print(f"Nome: {self.collection_name}")
        print(f"Total de Documentos: {stats['data']['entityCount']}")
        print(f"Dimensão dos Vetores: {stats['data']['dimension']}")
        print(f"Métrica de Similaridade: {stats['data']['metricType']}")
        return stats
    
    def analyze_text_data(self, sample_size=100):
        """Analisa os dados textuais da coleção"""
        print("\n🔍 Analisando dados textuais...")
        data = self.client.get_all_entities(self.collection_name)
        
        if not data:
            print("Nenhum dado encontrado na coleção.")
            return None
        
        # Criar DataFrame para análise
        df = pd.DataFrame(data)
        
        # Estatísticas básicas
        print(f"\n📝 Estatísticas Textuais (amostra de {len(df)} documentos):")
        print(f"Comprimento médio do texto: {df['text'].str.len().mean():.0f} caracteres")
        print(f"Comprimento mínimo: {df['text'].str.len().min()} caracteres")
        print(f"Comprimento máximo: {df['text'].str.len().max()} caracteres")
        
        # Distribuição por página (se existir o campo)
        if 'page' in df.columns:
            page_dist = Counter(df['page'])
            print("\n📖 Distribuição por Página:")
            for page, count in sorted(page_dist.items()):
                print(f"Página {page}: {count} documentos")
            
            # Plotar distribuição
            plt.figure(figsize=(10, 5))
            df['page'].value_counts().sort_index().plot(kind='bar')
            plt.title('Documentos por Página')
            plt.xlabel('Número da Página')
            plt.ylabel('Quantidade de Documentos')
            plt.tight_layout()
            plt.savefig('page_distribution.png')
            print("\n📈 Gráfico salvo como 'page_distribution.png'")
        
        return df
    
    def analyze_vectors(self, sample_size=10):
        """Analisa os vetores de embedding"""
        print("\n🧮 Analisando vetores de embedding...")
        data = self.client.query_entities(self.collection_name, limit=sample_size)
        
        if not data.get('data'):
            print("Nenhum dado encontrado na coleção.")
            return None
        
        vectors = [item['vector'] for item in data['data']]
        
        print("\n🔢 Estatísticas dos Vetores (amostra):")
        print(f"Dimensão: {len(vectors[0])} (esperado: 384)")
        print(f"Valor mínimo: {min(min(v) for v in vectors):.4f}")
        print(f"Valor máximo: {max(max(v) for v in vectors):.4f}")
        print(f"Valor médio: {sum(sum(v) for v in vectors)/(sample_size*len(vectors[0])):.4f}")
        
        return vectors
    
    def full_analysis(self):
        """Executa análise completa"""
        print(f"\n🔬 Iniciando análise da coleção '{self.collection_name}'")
        self.get_collection_stats()
        df = self.analyze_text_data()
        vectors = self.analyze_vectors()
        
        print("\n✅ Análise concluída!")
        return {
            "stats": self.client.get_collection_stats(self.collection_name),
            "text_samples": df.head(3).to_dict('records') if df is not None else None,
            "vector_samples": vectors[:2] if vectors is not None else None
        }

if __name__ == "__main__":
    analyzer = CollectionAnalyzer()
    analysis_results = analyzer.full_analysis()
    
    # Salvar resultados em JSON
    import json
    with open('collection_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print("\n📄 Resultados salvos em 'collection_analysis.json'")