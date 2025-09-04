# -*- coding: utf-8 -*-            
# @Time : 2025/9/4 23:49
# @Author : CodeDi
# @FileName: 三国蜀汉.py

from neo4j import GraphDatabase

# === 修改为你的 Neo4j 地址与密码 ===
uri = "bolt://127.0.0.1:7687"   # 如果是远程服务器，就改为 bolt://<服务器IP>:7687
user = "neo4j"
password = "Str0ngPass.147258"    


driver = GraphDatabase.driver(uri, auth=(user, password))

def create_shuhan_graph(driver):
    with driver.session(database="neo4j") as session:  # 默认数据库是 neo4j
        # 清空数据库，避免重复创建
        session.run("MATCH (n) DETACH DELETE n")

        # === 创建人物节点 ===
        persons = [
            {"name": "刘备", "title": "蜀汉昭烈帝"},
            {"name": "关羽", "title": "五虎上将"},
            {"name": "张飞", "title": "五虎上将"},
            {"name": "诸葛亮", "title": "丞相"},
            {"name": "赵云", "title": "五虎上将"},
            {"name": "黄忠", "title": "五虎上将"},
            {"name": "马超", "title": "五虎上将"},
        ]
        for p in persons:
            session.run(
                "CREATE (:Person {name:$name, title:$title})",
                name=p["name"], title=p["title"]
            )

        # === 创建关系 ===
        relations = [
            ("刘备", "关羽", "结义兄弟"),
            ("刘备", "张飞", "结义兄弟"),
            ("关羽", "张飞", "结义兄弟"),
            ("刘备", "诸葛亮", "君臣"),
            ("刘备", "赵云", "君臣"),
            ("刘备", "马超", "君臣"),
            ("刘备", "黄忠", "君臣"),
            ("诸葛亮", "赵云", "同僚"),
            ("诸葛亮", "关羽", "同僚"),
            ("诸葛亮", "张飞", "同僚"),
        ]
        for a, b, r in relations:
            session.run(
                """
                MATCH (p1:Person {name:$a}), (p2:Person {name:$b})
                CREATE (p1)-[:RELATION {type:$r}]->(p2)
                """,
                a=a, b=b, r=r
            )

        print("✅ 蜀汉人物关系图谱已创建完成！")

if __name__ == "__main__":
    create_shuhan_graph(driver)
    driver.close()
