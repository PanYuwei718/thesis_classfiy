import mysql.connector
import numpy as np
import csv

def get_doc_score_pair(lecture_num = 0,question_no = 0):
    all_kbn_list =  ["globalishizawa", "scienceishizawa", "easiaishizawa", "criticizeishizawa"]
    all_question_no_list = ["1","2","3"]
    lecture_kbn_list = []
    question_no_list = []
    doc_score_list = np.empty((0,2))
    lecture_num = int(lecture_num)
    question_no = int(question_no)
    print(lecture_num)
    if lecture_num < 0 and -5 < lecture_num: #対応するドキュメントは取ってこない
        lecture_kbn_list = all_kbn_list
        lecture_kbn_list.pop(-(lecture_num))
        print(lecture_kbn_list)
    elif lecture_num > 0 and lecture_num < 5: #対応するドキュメントだけ取ってくる
        lecture_kbn_list.append(all_kbn_list[lecture_num])
        print(lecture_kbn_list)
    else:
        lecture_kbn_list = all_kbn_list #全てのドキュメントを取ってくる
        print(lecture_kbn_list)

    if question_no < 0 and -4 < question_no: #対応するドキュメントは取ってこない
        question_no_list = all_question_no_list
        question_no_list.pop(-(question_no))
        print(question_no_list)
    elif question_no > 0 and question_no < 4:
        question_no_list.append(all_question_no_list[question_no])
        print(question_no_list)
    else:
        question_no_list = all_question_no_list
        print(question_no_list)

    conn = mysql.connector.connect(
        host = 'localhost',
        port =  3306,
        user = 'panyuwei',
        password = 'WorkAB$123#',
        database ='ases2020pan'
    )

    # print(conn.is_connected())  #=> Bool 繋がるかどうか

    cur = conn.cursor()
    a = []
    for lecture in lecture_kbn_list:
        for qustion in question_no_list:
            cur.execute("SELECT answer_body,jess1 FROM t_answer WHERE lecture_kbn=%s AND question_no=%s",
                        (lecture, qustion,))
            rows = cur.fetchall()
            for doc_score in rows:
                print(doc_score)
                try:
                    doc_score_list = np.append(doc_score_list, np.array(
                        [[str(doc_score[0].replace("\u3000", "").replace("\n", "")), float(doc_score[1])]]), axis=0)
                except:
                    # print("-----------",doc_score[0])＃何もない値があるのでスルーしてもらう
                    pass
                
                #print(doc_score_list.shape)
                #print(len(doc_score_list))
                #return doc_score_list  # 帰ってくるのは行数が小論文の数で２列の配列
            return doc_score_list



def main():
    mylist = get_doc_score_pair(1, 1)
    with open('data_1_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in mylist:
            writer.writerow(row)


if __name__ == "__main__":
    main()
