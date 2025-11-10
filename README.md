# [Python] Patent-Analysis

🗓️ **Date**: 2024.11.21 ~ 2025.07.09

<br/>

📊 **Objective**
 1. Collection of USPTO patent data using a web crawler.
 2. Propose new product idea discovery by applying NLP to patent abstract data.
<br/>

🧩 **Table of Contents**
|Num|Content|
|----|-------|
|01|[Crawling Data](https://github.com/git-jihyunpark/Patent-Analysis/blob/main/1_crawling_patents_year.ipynb)|
|02|[Data Preprocessing](https://github.com/git-jihyunpark/Patent-Analysis/blob/main/2_data_preprocessing.ipynb)|
|03|[LDA](https://github.com/git-jihyunpark/Patent-Analysis/blob/main/3_LDA.ipynb)
|04|[NuNER](https://github.com/git-jihyunpark/Patent-Analysis/blob/main/4_NuNER.ipynb)|
<br/>


## 🔷 Project: New product idea discovery using LDA and NER


📌 **Introduction**
- Since morphology analysis is mainly conducted based on expert knowledge, it tends to be subjective or biased.
- To overcome this limitation, a data-driven methodology for constructing morphology analysis is proposed.
- Collected patent data related to `wearable devices` and constructed a morphological matrix using LDA and NuNER.
- Proposed new product ideas through morphology analysis.
<br/>


📂 **Dataset**
- USPTO Data Collection
  - Period: 2023.01.01 – 2023.12.31
  - Condition: Patent abstracts containing the keyword `wearable device`
  - Collection Results: A total of 10,672
<br/>

### 1. Crawling Data

- Set data collection criteria on Google Patents, retrieve the data, and download the patent numbers as a CSV file.
![image](https://github.com/user-attachments/assets/553e41b0-6ad6-4e3c-9ad4-120f697d570e)
<br/><br/>


- Read a CSV file containing a list of patent numbers and crawl detailed information for each patent from Google Patents.<br/> 
```python
# 특허 번호에 대한 크롤링
def crawling_patents(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv, skiprows=1)
    df['patent_number'] = df['id'].apply(lambda x: f"US{x.split('-')[1]}B2" if pd.notnull(x) else None)
    df = df[df['patent_number'].notnull()].reset_index(drop=True)

    results = []
    for pat in df['patent_number']:
        data = fetch_google_patent(pat)
        if data:
            results.append(data)
        time.sleep(1)  # polite crawling

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Data saved to {output_csv}")
    else:
        print("⚠️ No data collected.")
```
<br/><br/>


- Automatically collect patent information (title, abstract, filing date, grant date, etc.) from Google Patents based on a specific patent number. <br/>
```python
# Google Patents 정보 수집
def fetch_google_patent(patent_number):
    url = f'https://patents.google.com/patent/{patent_number}/en'
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"[{patent_number}] ❌ Status Code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    try:
        title_tag = soup.find('meta', {'name': 'DC.title'})
        abstract_tag = soup.find('meta', {'name': 'DC.description'})
        filed_tag = soup.find('meta', {'scheme': 'dateFiled'})
        granted_tag = soup.find('meta', {'scheme': 'dateIssued'})

        title = title_tag['content'] if title_tag else None
        abstract = abstract_tag['content'] if abstract_tag else None
        date_filed = filed_tag['content'] if filed_tag else None
        date_granted = granted_tag['content'] if granted_tag else None

        if not title and not abstract:
            print(f"[{patent_number}] ❌ Empty content. Skipped.")
            return None

        return {
            'patent_number': patent_number,
            'title': title,
            'abstract': abstract,
            'date_filed': date_filed,
            'date_granted': date_granted
        }

    except Exception as e:
        print(f"[{patent_number}] ❌ Parsing Error: {e}")
        return None
```
<br/><br/>


- Save patent information (title, abstract, filing date, grant date, etc.) as a CSV file. <br/>
```python
crawling_patents(
    input_csv='Data/gp-search-20250611-235346.csv',
    output_csv='wearable_devices_patents_2023.csv'
)
```
<br/><br/>


### 2. Data Preprocessing
- After reading the CSV file, apply preprocessing to the `abstract` column. <br/>
  - Convert to lowercase, remove punctuation and numbers, perform tokenization, remove stopwords, and apply lemmatization.
```python
def preprocess_text(file_path, output_file):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 데이터 프레임의 첫 몇 줄 확인
    print(f"Processing {file_path}")
    print(df.head())

    # 전처리 함수
    def clean_text(text):
        # 텍스트가 문자열인지 확인하고 문자열이 아닌 경우 빈 문자열로 변환
        if not isinstance(text, str):
            text = ""

        # 소문자로 변환
        text = text.lower()
        # 구두점 및 숫자 제거
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # 토큰화
        tokens = text.split()
        # 불용어 제거
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # 표제어 추출
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    # 'abstract' 열에 전처리 적용
    if 'abstract' in df.columns:        
        df['cleaned_abstract'] = df['abstract'].apply(clean_text)
    else:
        print(f"'abstract' 열이 {file_path}에 존재하지 않습니다.")

    # 전처리된 데이터프레임을 CSV 파일로 저장
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Processed data saved to {output_file}")
    
    # 전처리 완료된 데이터프레임 반환
    return df
```
<br/><br/>

- Save the preprocessing results as a CSV file. <br/>
```python
df_2023 = preprocess_text('Data/wearable_devices_patents_2023.csv', 'Data/wearable_devices_processed_2023.csv')
```
<br/><br/>



### 3. LDA
- Extracted topics from the data using the LDA model. <br/>
```python
def lda_modeling(cleaned_abstracts):
    # TF-IDF를 사용하여 텍스트 데이터 벡터화
    vectorizer = TfidfVectorizer(max_df=0.75, min_df=5, stop_words='english')
    X = vectorizer.fit_transform(cleaned_abstracts)

    # 데이터 벡터화된 것을 gensim에서 사용할 수 있도록 변환
    corpus = Sparse2Corpus(X, documents_columns=False)
    id2word = {i: token for i, token in enumerate(vectorizer.get_feature_names_out())}

    # Perplexity와 Coherence 저장할 리스트 초기화
    perplexity_scores = []
    coherence_scores = []
    topic_range = range(5, 31, 1)  # 5에서 30까지 1단위로 토픽 수 조정

    # 각 토픽 수에 따른 LDA 모델 학습 및 평가
    for num_topics in topic_range:
        # sklearn LDA 모델 학습
        lda_model = LatentDirichletAllocation(n_components=num_topics, learning_decay=0.7, random_state=42)
        lda_model.fit(X)

        # Perplexity 계산 (scikit-learn의 LDA 모델로 계산)
        perplexity = lda_model.perplexity(X)
        perplexity_scores.append(perplexity)

        # Gensim의 LDA 모델을 사용해 Coherence 계산
        gensim_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            passes=10,
            iterations=50,
            random_state=42,
            alpha='auto'
        )

        # Coherence 계산
        coherence_model = CoherenceModel(
            model=gensim_model,
            texts=[doc.split() for doc in cleaned_abstracts],  # 'cleaned_abstracts'를 텍스트로 변환
            dictionary=Dictionary.from_corpus(corpus, id2word),
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        coherence_scores.append(coherence)

    # 결과 반환
    return perplexity_scores, coherence_scores, X, vectorizer
```
<br/><br/>

- Calculated Perplexity and Coherence to determine the optimal number of LDA topics and visualized the results in a graph. <br/>
```python
def lda_plot(perplexity_scores, coherence_scores):
    # 토픽 범위 (5에서 30까지 1단위로 토픽 수 조정)
    topic_range = range(5, 31, 1)
    
    # 그래프 크기 설정
    plt.figure(figsize=(12, 5))
    
    # Perplexity 그래프
    plt.subplot(1, 2, 1)
    plt.plot(list(topic_range), perplexity_scores, marker='o', color='blue')
    plt.title('Perplexity of LDA Models')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    for i, txt in enumerate(perplexity_scores):
        plt.annotate("", (topic_range[i], perplexity_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
        # Uncomment to show Perplexity scores as text on graph
        # plt.annotate(f"{txt:.1f}", (topic_range[i], perplexity_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Coherence 그래프
    plt.subplot(1, 2, 2)
    plt.plot(list(topic_range), coherence_scores, marker='o', color='red')
    plt.title('Coherence Score by Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    for i, txt in enumerate(coherence_scores):
        plt.annotate("", (topic_range[i], coherence_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
        # Uncomment to show Coherence scores as text on graph
        # plt.annotate(f"{txt:.2f}", (topic_range[i], coherence_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 그래프 레이아웃 조정 및 출력
    plt.tight_layout()
    plt.show()
```
<br/><br/>

- The number of LDA topics was determined as 8, 12, and 16 based on the following criteria:
  - The number of components in wearable devices
  - The complexity of the model
  - The trade-off relationship between perplexity and coherence
<img width="1189" height="490" alt="LDA" src="https://github.com/user-attachments/assets/f9cdff22-af59-4f7e-883f-06b7c6d5f916" />
<br/><br/>


- Extracted top keywords for each number of topics.
```python
# 토픽 수
topics_8 = 8

lda = LatentDirichletAllocation(n_components=topics_8, random_state=42)
lda.fit(X_2023)

topics = display_topics(lda, vectorizer_2023.get_feature_names_out(), num_top_words)

# 결과를 데이터프레임으로 변환 (topic, keywords 두 컬럼)
df_topics_8 = pd.DataFrame(list(topics.items()), columns=['topic', 'keywords'])

# 결과 엑셀 파일로 저장
output_file = 'Data/wearable_devices_2023_lda_topic_8.csv'
df_topics_8.to_csv(output_file, index=False)

# 결과 출력
for topic_num, words in topics.items():
    print(f"{topic_num}: {words}\n")
```
<br/><br/>


- Reviewed the keywords and defined the `dimensions` of the morphological matrix based on expert knowledge.
```python
# 각 토픽별 dimension 정의(전문가 지식 기반)
labels_8 = {
    'Topic 0': 'Data Processing',
    'Topic 1': 'Wireless Communication',
    'Topic 2': 'User Interface',
    'Topic 3': 'Audio System',
    'Topic 4': 'Optical Technology',
    'Topic 5': 'Materials',
    'Topic 6': 'Network System',
    'Topic 7': 'Sensors'
}

# CSV 파일 읽기
df_2023_topic_8 = pd.read_csv('Data/wearable_devices_2023_lda_topic_8.csv')

# label 추가
df_2023_topic_8['label'] = df_2023_topic_8['topic'].map(labels_8)

# 컬럼 순서 'label', 'topic', 'keywords' 순으로 변경
df_2023_topic_8 = df_2023_topic_8[['label', 'topic', 'keywords']]

# 결과 엑셀 파일로 저장
output_file = 'Data/wearable_devices_2023_lda_topic_8_with_labels.csv'
df_2023_topic_8.to_csv(output_file, index=False)

# 결과 확인
df_2023_topic_8
```
<br/><br/>




### 4. NuNER

```python

```
<br/><br/>



```python

```
<br/><br/>





---


💖 **Lesson & Learn**
1. Improvement of data collection and NLP 
   > USPTO patent data <br/>
   > LDA, NuNER
2. Discovery of new product ideas  
   > Discovery of new product ideas based on morphological analysis






