from pymilvus import MilvusClient, DataType

def create_diary_schema():
    """Cria schema otimizado para o diário do Dr. Voss"""
    schema = MilvusClient.create_schema(
        auto_id=False,  # Vamos gerar os IDs manualmente
        enable_dynamic_field=True  # Permite campos adicionais não definidos
    )
    
    # 1. Campo Primário (Identificador Único)
    schema.add_field(
        field_name="entry_id",
        datatype=DataType.VARCHAR,
        max_length=64,
        is_primary=True
    )
    
    # 2. Campo Vetorial (Embeddings)
    schema.add_field(
        field_name="content_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=384  # Dimensão do Snowflake Arctic Embed
    )
    
    # 3. Campos Escalares (Metadados Essenciais)
    schema.add_field(
        field_name="content_text",
        datatype=DataType.VARCHAR,
        max_length=65535  # Para textos longos
    )
    
    schema.add_field(
        field_name="entry_date",
        datatype=DataType.VARCHAR,
        max_length=64  # Formato: "7th Day of Snowrest 1856"
    )
    
    schema.add_field(
        field_name="entry_title",
        datatype=DataType.VARCHAR,
        max_length=256
    )
    
    schema.add_field(
        field_name="day_number",
        datatype=DataType.INT16
    )
    
    schema.add_field(
        field_name="month",
        datatype=DataType.VARCHAR,
        max_length=32
    )
    
    schema.add_field(
        field_name="year",
        datatype=DataType.INT16
    )
    
    schema.add_field(
        field_name="paragraph_number",
        datatype=DataType.INT16
    )
    
    schema.add_field(
        field_name="word_count",
        datatype=DataType.INT16
    )
    
    schema.add_field(
        field_name="line_count",
        datatype=DataType.INT16
    )
    
    schema.add_field(
        field_name="is_date_entry",
        datatype=DataType.BOOL
    )
    
    return schema