from . import pandas_utils
import pandas as pd

def setup_table(conn, table_name, column_names, column_types, charset="utf8", auto_increment=1):
    cursor = conn.cursor()
    columns = ""
    for column_name, column_type in zip(column_names, column_types):
        columns += f"`{column_name}` {column_type},\n\t"
    columns = columns[:-3]

    sql_commands = f"""
    DROP TABLE IF EXISTS `{table_name}`;
    
    CREATE TABLE IF NOT EXISTS `{table_name}` (
    {columns}
    ) DEFAULT CHARSET={charset} AUTO_INCREMENT={auto_increment};
    """
    
    for command in sql_commands.strip().split(';'):
        if command.strip():
            cursor.execute(command)
    
    conn.commit()
    print(f"Table `{table_name}` has been created successfully in the database.")
    cursor.close()


def add_key_to_table(conn, table_name, key):
    cursor = conn.cursor()
    
    sql_commands = f"""
    ALTER TABLE `{table_name}`
    ADD PRIMARY KEY (`{key}`);
    """
    
    for command in sql_commands.strip().split(';'):
        if command.strip():
            cursor.execute(command)
    
    conn.commit()
    print(f"Table `{table_name}` now has primary key `{key}`.")
    cursor.close()


def upload_csv_to_table(conn, table_name, file_name):
    cur = conn.cursor()
    cur.execute(f"LOAD DATA LOCAL INFILE '{file_name}' INTO TABLE `{table_name}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()


def upload_data_from_file(conn, file, table_name, type, key):
    print(f"Uploading `{file}` to table `{table_name}`...")
    file_df = pandas_utils.load_csv(file)
    
    setup_table(conn, table_name, file_df.columns, type)
    add_key_to_table(conn, table_name, key)
    upload_csv_to_table(conn, table_name, file)

    print(f"Uploaded `{file}` to table `{table_name}` successfully!")


def query_AWS_load_table(conn, table_name, columns=None):
    if columns is None:
        query_str = f"SELECT * FROM {table_name};"
    else:
        cols = ", ".join(columns)
        query_str = f"SELECT {cols} FROM {table_name};"
    
    cur = conn.cursor()
    
    cur.execute(query_str)
    data = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    
    df = pd.DataFrame(data, columns=colnames)
    return df


def upload_data_from_df(conn, df, table_name, types, key):
    columns = "".join([f"`{field}` {t},\n" for field, t in zip(df.columns, types)])[:-2]

    setup_table(conn, table_name, columns, types)
    add_key_to_table(conn, table_name, key)

    df.to_csv("upload_temp.csv", index=False)
    upload_csv_to_table(conn, table_name, "upload_temp.csv")
