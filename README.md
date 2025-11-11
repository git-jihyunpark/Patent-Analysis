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

- The number of LDA topics was determined as `8`, `12`, and `16` based on the following criteria:
  - The number of components in wearable devices
  - The complexity of the model
  - The trade-off relationship between perplexity and coherence
<img width="1189" height="490" alt="LDA" src="https://github.com/user-attachments/assets/f9cdff22-af59-4f7e-883f-06b7c6d5f916" />
<br/><br/>


- Extracted top keywords for each number of topics.(Conducted the same process for topic numbers 12 and 16.)
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


- Reviewed the keywords and defined the `dimensions` of the morphological matrix based on expert knowledge.(Conducted the same process for topic numbers 12 and 16.)
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
```
<br/><br/>




### 4. NuNER

- NuNER, a model pre-trained on LLM-annotated data across various domains to overcome the limitations of traditional NER, is well-suited for extracting diverse and creative values in morphology analysis (Bogdanov et al., 2024).<br/>
```python
from gliner import GLiNER
model = GLiNER.from_pretrained("numind/NuZero_token")
```
<br/><br/>


- Extracted entities using NuNER for each defined dimension.(Conducted the same process for topic numbers 12 and 16.)<br/>
```python
# 추출 결과를 저장할 리스트
results_2023_topic_8_entity = []

# 각 텍스트에 대해 엔티티 추출
for tokens in unique_df['tokens']:
    if isinstance(tokens, list):
        text = ' '.join(tokens)
    else:
        text = str(tokens)

    entities = model.predict_entities(text, label_2023_topic_8)

    for entity in entities:
        #print(entity["text"], "=>", entity["label"])
        results_2023_topic_8_entity.append({
            'Text': entity['text'],
            'Label': entity['label']
        })

# 결과 리스트를 데이터 프레임으로 변환
df_2023_topic_8_entity = pd.DataFrame(results_2023_topic_8_entity)

# Label별로 Text를 합치기
df_2023_topic_8_grouped = df_2023_topic_8_entity.groupby('Label')['Text'].apply(lambda x: ', '.join(x)).reset_index()

# 결과 데이터 프레임 출력
df_2023_topic_8_grouped.head()
```
<br/><br/>



### 5. Morphological Analysis
- Compared the extracted dimensions for each number of topics to select those that could serve as components of wearable devices. <br/>

*Table. Result of Topic 8 LDA: Keywords and Dimension Definitions* <br/>
  
| **Topic** | **Keywords** | **Corresponding Dimension** |
|------------|--------------|------------------------------|
| **Topic 0** | data, user, device, content, image, method, based, object, information, second, plurality, model, associated, medium, set, application, video, vehicle, item, computing | Data Processing |
| **Topic 1** | ue, wireless, resource, signal, transmission, channel, communication, information, second, beam, configuration, station, device, uplink, radio, power, base, control, cell, frequency | Wireless Communication |
| **Topic 2** | display, device, object, image, camera, user, screen, electronic, virtual, second, position, input, gesture, haptic, wearable, processor, sensor, eye, hand, area | User Interface |
| **Topic 3** | block, current, prediction, motion, transform, image, decoding, picture, sample, basis, deriving, vector, residual, mode, information, intra, bitstream, based, coefficient, pressure | Audio System |
| **Topic 4** | audio, sound, signal, microphone, sequence, acoustic, speaker, speech, clip, ear, guest, hearing, transcription, bit, firsttype, quantum, binaural, frame, sporting, music | Optical Technology |
| **Topic 5** | surface, housing, portion, second, lens, assembly, material, element, device, structure, display, configured, layer, member, flexible, includes, disposed, electronic, conductive, component | Materials |
| **Topic 6** | device, network, communication, data, user, access, request, service, information, second, transaction, method, associated, message, application, wireless, identifier, key, server, packet | Network System |
| **Topic 7** | signal, light, circuit, pixel, display, second, voltage, layer, electrode, configured, line, power, plurality, includes, device, sensor, driving, touch, sensing, region | Sensors |

<br/><br/>


*Table. Result of Topic 12 LDA: Keywords and Dimension Definitions* <br/>
  
| **Topic** | **Keywords** | **Corresponding Dimension** |
|------------|--------------|------------------------------|
| **Topic 0** | user, data, device, image, content, method, based, object, second, information, model, associated, video, plurality, medium, application, item, set, input, event | Data Processing |
| **Topic 1** | wireless, ue, resource, communication, transmission, signal, information, channel, device, second, station, configuration, method, radio, beam, base, control, uplink, network, power | Wireless Communication |
| **Topic 2** | animation, trip, audience, nft, flight, error, fov, telematics, dispensing, dispenser, dialog, fuel, exercise, scell, defect, welding, hazard, signature, meter, particle | Applications |
| **Topic 3** | block, current, prediction, transform, decoding, picture, sample, basis, motion, deriving, vector, residual, pressure, intra, air, bitstream, mode, information, heat, flag | Sensors |
| **Topic 4** | image, audio, signal, device, object, user, virtual, sensor, wearable, sound, position, display, camera, eye, environment, second, configured, movement, head, reality | User Interface |
| **Topic 5** | insurance, drone, headworn, transportation, vehicle, parking, sentiment, av, trp, observed, pickup, playlist, mb, door, compartment, charging, shadow, selfdriving, chassis, pll | External Devices |
| **Topic 6** | device, data, network, user, memory, access, request, communication, service, information, second, method, associated, application, key, computing, server, transaction, storage, based | Network System |
| **Topic 7** | circuit, signal, pixel, voltage, second, display, power, electrode, driving, line, transistor, configured, layer, panel, output, connected, plurality, includes, touch, control | Display |
| **Topic 8** | utterance, speech, audio, conference, topic, voice, earpiece, spoken, chatbot, assistant, speaker, transcription, transcript, satellite, conferencing, disease, portal, post, backup, container | Audio System |
| **Topic 9** | light, display, surface, second, layer, portion, housing, optical, lens, substrate, disposed, element, area, region, includes, device, structure, configured, material, sensor | Optical Technology |
| **Topic 10** | compound, clip, ligand, atm, formula, episode, shooting, stroke, carbon, rlm, polar, strand, binding, printer, fixing, mo, mixture, lowpower, neighbor, ecc | Materials |
| **Topic 11** | wtru, memory, bwp, cell, bit, repeater, reservoir, water, tensor, fan, read, pump, duty, codeword, filling, uci, branch, gear, quantum, po | Memory |

<br/><br/>


*Table. Result of Topic 16 LDA: Keywords and Dimension Definitions* <br/>

| **Topic** | **Keywords** | **Corresponding Dimension** |
|------------|--------------|------------------------------|
| **Topic 0** | user, data, image, device, content, object, method, based, second, display, model, information, video, input, plurality, medium, virtual, interface, associated, item | User Interface |
| **Topic 1** | resource, sidelink, signal, symbol, slot, sr, information, second, reference, channel, method, related, time, sl, physical, timefrequency, window, wireless, set, rach | Wireless Signal Setup |
| **Topic 2** | voltage, memory, cell, word, read, envelope, sensory, inductor, barcode, converter, diffusion, regulator, venue, resistor, bit, line, error, bias, amplifier, page | Circuit |
| **Topic 3** | patient, treatment, signal, sensor, medical, surgical, temperature, rate, blood, data, monitoring, robot, measurement, pressure, method, heart, based, plan, skin, therapy | Medical Devices |
| **Topic 4** | audio, sound, acoustic, signal, haptic, speaker, microphone, hand, ear, user, wave, computergenerated, tissue, feedback, headworn, transducer, food, head, modality, reflective | Audio System |
| **Topic 5** | power, signal, antenna, circuit, frequency, configured, rf, second, vibration, charging, battery, wireless, energy, device, cable, band, accessory, coil, switch, control | Power |
| **Topic 6** | device, network, data, user, communication, service, access, associated, request, information, application, method, transaction, computing, second, message, mobile, key, based, identifier | Network System |
| **Topic 7** | pixel, display, circuit, second, signal, voltage, light, line, driving, electrode, element, layer, transistor, panel, lens, plurality, region, connected, driver, gate | Display |
| **Topic 8** | data, memory, storage, device, file, card, request, host, second, controller, address, operation, plurality, information, command, record, user, server, service, table | Memory |
| **Topic 9** | block, current, prediction, decoding, transform, sample, picture, bit, basis, deriving, residual, vector, flag, intra, bitstream, information, mode, coefficient, motion, encoding | Data Processing |
| **Topic 10** | emissive, cleaning, retention, parking, deposit, documentation, oled, spot, print, fund, bond, clinical, velocity, blind, rider, resonance, damaged, household, compilation, grip | Display Manufacturing |
| **Topic 11** | die, test, clock, ray, clip, bank, primitive, guest, comment, wheel, footwear, semiconductor, register, interconnect, intersection, comparator, tunnel, rating, dy, belt | Semiconductor |
| **Topic 12** | vehicle, emergency, financial, safety, handheld, biometric, person, telephone, marker, chatbot, chain, post, building, assistance, occupant, fault, ultrasonic, caller, lidar, fragment | Emergency |
| **Topic 13** | ue, wireless, communication, channel, transmission, device, information, signal, station, second, resource, beam, configuration, network, control, base, cell, method, uplink, aspect | Wireless Communication |
| **Topic 14** | surface, display, second, housing, portion, light, layer, device, optical, sensor, substrate, structure, configured, disposed, material, electronic, includes, assembly, area, touch | Sensors |
| **Topic 15** | vehicle, autonomous, browser, operator, collision, web, road, listing, uci, trip, reservation, kiosk, fitness, exit, dispenser, desktop, cache, repair, fleet, launch | External Devices |

<br/><br/>



- Selected values from NuNER’s entity lists corresponding to the chosen dimensions that could represent potential components of wearable devices. <br/>

*Table. Results of NuNER Entity extraction for Topic 8* <br/>

| **Topic** | **Dimension** | **Entities** |
|------------|----------------|---------------|
| **Topic 0** | Data Processing | bytestream, datastreaming, datahandling, pipeline, beamforming, columnar, coding, dataflow, lineage, computation, subsampling, informationhandling, dtree, kmeans, datawriting, classifier, defragmenting, packetization, fingerprinting, cloudcomputing, dataparallel, datatransfer, recognition, analytics, digitally, demultiplexing |
| **Topic 1** | Wireless Communication | wlan, bluetooth, vxlan, wifi |
| **Topic 2** | User Interface | ui, touchscreen, satellite, usermenu, teleprompter, toolbar, gui, microvisor, uxui, lcd, microdisplay, touchscreenbased, iframe, touchpad, trackpad, headsupdisplay, hud, forcetouch, dualscreen |
| **Topic 3** | Audio System | earbuds, earbud, earphone, microphone, speakerphone, headphone |
| **Topic 4** | Optical Technology | nearinfrared, waveguide, fiberoptic, lightscattering, electrooptical |
| **Topic 5** | Materials | iridium, hexylammonium, graphite, glassceramic, polyurethane, copper, aluminum, vanadium, silicate, nickel, resin, cellulose, alumina, garnet, gallium, barium, nanosheets, graphene, phosphide, nanofibers, polyester, carbon, titanate, ceramic, fiber, aramid, polyethylene, glass, silicone, silver, perovskites, silicon |
| **Topic 6** | Network System | ethernet |
| **Topic 7** | Sensors | nanosensors, photosensors, biosensors, photosensor |

<br/><br/>


*Table. Results of NuNER Entity extraction for Topic 12* <br/>

| **Topic** | **Dimension** | **Entities** |
|------------|----------------|---------------|
| **Topic 0** | Data Processing | bytestream, datastreaming, datahandling, beamforming, columnar, coding, dataflow, computation, subsampling, informationhandling, dtree, cachelines, kmeans, datawriting, classifier, defragmenting, hashing, packetization, fingerprinting, cloudcomputing, dataparallel, datatransfer, recognition, analytics, demultiplexing |
| **Topic 1** | Wireless Communication | wlan, bluetooth, vxlan, wifi |
| **Topic 2** | Applications | watch |
| **Topic 3** | Sensors | nanosensors, photosensors, biosensors, photosensor |
| **Topic 4** | User Interface | touchscreen, satellite, usermenu, toolbar, gui, microvisor, uxui, interface, iframe |
| **Topic 5** | External Devices | smartwatches |
| **Topic 6** | Network System | ethernet |
| **Topic 7** | Display | microdisplays, lcd, microdisplay |
| **Topic 8** | Audio System | headphone |
| **Topic 9** | Optical Technology | lightscattering |
| **Topic 10** | Materials | iridium, hexylammonium, graphite, glassceramic, aluminum, vanadium, silicate, nickel, resin, cellulose, garnet, nanosheets, graphene, phosphide, nanofibers, polyester, ceramic, aramid, polyethylene, silver, perovskites, silicon |
| **Topic 11** | Memory | multimemory, mem, mempool, gesturebased, memory, intermemory, inmemory |

<br/><br/>


*Table. Results of NuNER Entity extraction for Topic 16* <br/>

| **Topic** | **Dimension** | **Entities** |
|------------|----------------|---------------|
| **Topic 0** | User Interface | touchscreen, usermenu, toolbar, headupdisplays, gui, microvisor, displaypanel, iframe |
| **Topic 1** | Wireless Signal Setup | - |
| **Topic 2** | Circuit | udi, circuit, capacitor, subcircuits, circuitry |
| **Topic 3** | Medical Devices | inktoner, condom, cigarette, earbuds, oximeter, tidal, inverter, syringe, piston, catheter, endoscope, biologics, earbud, microneedles, microphone, electrocardiograph, orthotic, orthotics, airbag, inhaler, prosthetics, wristband, defibrillator, inkjet, binoculars, smartwatches, cardioverter, microendoscopes, flowmeters, microneedle, disability, dispenser, conditioner, lotion, airbags, eyeware, cellphone, cannula, spectacle, toothbrush, supraaural, mattress |
| **Topic 4** | Audio System | headphone |
| **Topic 5** | Power | powersupply |
| **Topic 6** | Network System | - |
| **Topic 7** | Display | lcd |
| **Topic 8** | Memory | multimemory, mem, mempool, gesturebased, memory, intermemory, inmemory |
| **Topic 9** | Data Processing | bytestream, datastreaming, datahandling, beamforming, columnar, coding, dataflow, lineage, computation, subsampling, informationhandling, kmeans, datawriting, classifier, defragmenting, packetization, fingerprinting, cloudcomputing, dataparallel, datatransfer, analytics, preprocessing, demultiplexing |
| **Topic 10** | Display Manufacturing | - |
| **Topic 11** | Semiconductor | iridium, hexylammonium, graphite, subtransistor, silyl, qci, resistor, supercapacitors, postsilicon, thinfilmtransistor, tungsten, phototransistor, crosscarrier, anode, silicate, acrylamidomethylpropanesulfonic, microsemiconductor, alumina, ammonium, garnet, mxene, thioxanthene, quartz, chromium, manganese, arsenide, ssd, indium, onchip, aryloxy, triphenylene, boron, gallium, barium, nanosheets, graphene, ammonia, germanium, perovskite, molybdenum, phosphide, titanium, mtc, carbene, cycloalkenyl, aramid, polyethylene, carbazolylcarbazole, polysilicon, neodymium, perovskites |
| **Topic 12** | Emergency | - |
| **Topic 13** | Wireless Communication | wlan, bluetooth, wifi |
| **Topic 14** | Sensors | nanosensors, photosensors, silicon |
| **Topic 15** | External Devices | - |
  
<br/><br/>


- Organized the selected dimensions and values into rows and columns to construct a morphological matrix.<br/>
  - The morphological matrix consists of a total of 16 dimensions. In particular, the dimensions `G (Materials)`, `H (Medical Devices)`, and `J (Semiconductor)` yielded more than 20 combinable values, demonstrating that a data-driven approach can effectively extend and complement expert judgment in constructing the morphological matrix.
  - Conversely, the dimensions `A (Applications)`, `E (External Devices)`, `K (Network System)`, and `M (Power)` showed a relatively limited number of extracted values. This limitation is interpreted as being caused by the issue of data sparsity, where relevant keywords are either absent or infrequently occurring in the collected dataset, similar to the previous case study.
- Combined values within the morphological matrix to discover new product ideas.<br/>

*Table. Morphological matrix for Wearable Devices* <br/>

| **(A) Applications** | **(B) Audio System** | **(C) Circuit** | **(D) Display** | **(E) External Devices** |
|------------------------|----------------------|------------------|------------------|----------------------------|
| A₁ = watch | B₁ = earbud | C₁ = capacitor | D₁ = LCD (Liquid Crystal Display) | E₁ = smart watches |
|  | B₂ = earphone | C₂ = circuitry | D₂ = micro-display |  |
|  | B₃ = microphone | C₃ = subcircuits |  |  |
|  | B₄ = speakerphone | C₄ = UDI (Unique Device Identifier) |  |  |
|  | B₅ = headphone |  |  |  |


| **(F) Data Processing** | **(G) Materials** | **(H) Medical Devices** | **(I) Memory** | **(J) Semiconductor** |
|--------------------------|-------------------|--------------------------|----------------|------------------------|
| F₁ = analytics | G₁ = alumina | H₁ = airbag | I₁ = gesture-based | J₁ = acrylamido methylpropane sulfonic |
| F₂ = beamforming | G₂ = aluminum | H₂ = binoculars | I₂ = In Memory | J₂ = ammonia |
| F₃ = byte-stream | G₃ = aramid | H₃ = cannula | I₃ = inter memory | J₃ = ammonium |
| F₄ = cachelines | G₄ = barium | H₄ = cardioverter | I₄ = mem pool (Memory Pool) | J₄ = anode |
| F₅ = classifier | G₅ = carbon | H₅ = catheter | I₅ = multi-memory | J₅ = aramid |
| F₆ = cloud computing | G₆ = cellulose | H₆ = defibrillator |  | J₆ = arsenide |
| F₇ = columnar | G₇ = ceramic | H₇ = dispenser |  | J₇ = aryloxy |
| F₈ = dataflow | G₈ = copper | H₈ = electro-cardiograph |  | J₈ = barium |
| F₉ = data handling | G₉ = fiber | H₉ = endoscope |  | J₉ = boron |
| F₁₀ = data parallel | G₁₀ = gallium | H₁₀ = eye-ware |  | J₁₀ = carbazolyl carbazole |
| F₁₁ = data streaming | G₁₁ = garnet | H₁₁ = flowmeters |  | J₁₁ = carbene |
| F₁₂ = data transfer | G₁₂ = glass | H₁₂ = inhaler |  | J₁₂ = chromium |
| F₁₃ = data writing | G₁₃ = glass ceramic | H₁₃ = inverter |  | J₁₃ = cycloalkenyl |
| F₁₄ = defragmenting | G₁₄ = graphene | H₁₄ = mattress |  | J₁₄ = gallium |
| F₁₅ = demultiplexing | G₁₅ = graphite | H₁₅ = micro-endoscopes |  | J₁₅ = garnet |
| F₁₆ = finger printing | G₁₆ = hexylammonium | H₁₆ = micro-needle |  | J₁₆ = germanium |
| F₁₇ = hashing | G₁₇ = iridium | H₁₇ = orthotic |  | J₁₇ = graphene |
| F₁₈ = information handling | G₁₈ = nano-fibers | H₁₈ = oximeter |  | J₁₈ = graphite |
| F₁₉ = lineage | G₁₉ = nano-sheets | H₁₉ = prosthetics |  | J₁₉ = indium |
| F₂₀ = packetization | G₂₀ = nickel | H₂₀ = spectacle |  | J₂₀ = iridium |
| F₂₁ = recognition | G₂₁ = perovskites | H₂₁ = supra aural |  | J₂₁ = manganese |
| F₂₂ = subsampling | G₂₂ = phosphide | H₂₂ = syringe |  | J₂₂ = micro-semiconductor |
|  | G₂₃ = polyester | H₂₃ = tooth brush |  | J₂₃ = molybdenum |
|  | G₂₄ = polyethylene | H₂₄ = wristband |  | J₂₄ = MTC (Metal Top Contact) |
|  | G₂₅ = polyurethane |  |  | J₂₅ = mxene |
|  | G₂₆ = resin |  |  | J₂₆ = neodymium |
|  | G₂₇ = silicate |  |  | J₂₇ = on-chip |
|  | G₂₈ = silicon |  |  | J₂₈ = perovskite |
|  | G₂₉ = silicone |  |  | J₂₉ = phosphide |
|  | G₃₀ = silver |  |  | J₃₀ = photo-transistor |
|  | G₃₁ = titanate |  |  | J₃₁ = polyethylene |
|  | G₃₂ = vanadium |  |  | J₃₂ = polysilicon |
|  |  |  |  | J₃₃ = postsilicon |
|  |  |  |  | J₃₄ = quartz |
|  |  |  |  | J₃₅ = resistor |
|  |  |  |  | J₃₆ = silicate |
|  |  |  |  | J₃₇ = silyl |
|  |  |  |  | J₃₈ = SSD (Solid State Drive) |
|  |  |  |  | J₃₉ = sub transistor |
|  |  |  |  | J₄₀ = super capacitors |
|  |  |  |  | J₄₁ = thin film transistor |
|  |  |  |  | J₄₂ = thioxanthene |
|  |  |  |  | J₄₃ = titanium |
|  |  |  |  | J₄₄ = triphenylene |
|  |  |  |  | J₄₅ = tungsten |

| **(K) Network System** | **(L) Optical Technology** | **(M) Power** | **(N) Sensors** | **(O) UI (User Interface)** | **(P) Wireless Communication** |
|--------------------------|-----------------------------|----------------|-----------------|------------------------------|--------------------------------|
| K₁ = ethernet | L₁ = electro-optical | M₁ = power supply | N₁ = bio-sensors | O₁ = display panel | P₁ = bluetooth |
|  | L₂ = fiber-optic |  | N₂ = nano-sensors | O₂ = dual screen | P₂ = VXLAN (Virtual eXtensible LAN) (Network System) |
|  | L₃ = light-scattering |  | N₃ = photo-sensor | O₃ = force touch | P₃ = Wi-Fi |
|  | L₄ = near infrared |  |  | O₄ = GUI (Graphical User Interface) | P₄ = WLAN (Wireless Local Area Network) |
|  | L₅ = waveguide |  |  | O₅ = HUD (Heads-up Display) |  |
|  |  |  |  | O₆ = micro-visor |  |
|  |  |  |  | O₇ = teleprompter |  |
|  |  |  |  | O₈ = toolbar |  |
|  |  |  |  | O₉ = touch-pad |  |
|  |  |  |  | O₁₀ = touch-screen |  |
|  |  |  |  | O₁₁ = trackpad |  |
|  |  |  |  | O₁₂ = user menu |  |

 
---

✅ **References**
- Bogdanov, S., Constantin, A., Bernard, T., Crabbé, B., & Bernard, E. P. (2024, November). Nuner: Entity recognition encoder pre-training via llm-annotated data. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 11829-11841).
<br/><br/>

---


💖 **Lesson & Learn**
1. Improvement of Data Collection and NLP Skills 
   > USPTO patent data <br/>
   > LDA, NuNER
2. Discovery of New Product Ideas  
   > Discovery of new product ideas based on morphological analysis <br/>
   > Performed morphology analysis based on data to complement the limitations of expert knowledge-based approaches.






