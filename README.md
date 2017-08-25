# Google's *43 Rules of Machine Learning*

Перевод отличного руководства [Мартина Зинкевича](http://martin.zinkevich.org/)  ["Принципы машинного обучения"](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf), с дополнительными комментариями.

-----

Вы можете найти терминологию к этому руководству в файле  [terminology.md](/terminology.md).

Вы можете найти введение к этому руководству в файле [overview.md](/overview.md).

#### Содержание

1. [Before Machine Learning](#before-machine-learning)
2. [ML Phase 1: Your First Pipeline](#your-first-pipeline)
3. [ML Phase 2: Feature Engineering](#feature-engineering)
4. [ML Phase 3: Slow Growth, Optimation Refinement, and Complex Models](#slow-growth-and-optimization-and-complex-models)
5. [Related Work](/related_work.md)
6. [Acknowledgements & Appendix](/acknowledgements_and_appendix.md)

**Note**: *Asterisk* (\*) footnotes are my own. *Numbered* footnotes are Martin's.

## Прежде чем использовать машинное обучение

#### Правило #1: Не бойтесь запустить продукт без машинного обучения.*

Машинное обучение это круто, но оно требует данных. Теоретически, вы можете взять данные из другой проблемы, а затем настраивать модель под новый продукт, но это будет хуже работать базовых эвристик. Если вы думаете что машинное обучение даст вам 100 прирост, значит эвристика даст вам 50% тем же путем.

Например, вы ранжируете приложения в магазине,для этого вы должны использовать рейтинг или количество установок. Если вы обнаруживаете спам, фильтруйте отправителей, которые отправили спам до этого. Не бойтесь использовать правки человека. Если вам нужно упорядочить контакты, то отсортируйте их по частоте использования (или даже по алфавиту). Если машинное обучение не является абсолютно необходимым для вашего продукта, то не используйте его, пока не получите данные.


<sup>[Google Research Blog - The 280-Year-Old Algorithm Inside Google Trips](https://research.googleblog.com/2016/09/the-280-year-old-algorithm-inside.html?m=1)</sup>

#### Правило #2: Во-первых, спроектируйте и реализуйте метрики
До того как формализовать, что будет делать ваша система машинного обучения, узнайте как можно больше о вашей текущей системе. Сделайте это по следующим причинам: 
1. Легче получить разрешений от пользователей системы ранее
2. Если вы думаете, что что-то может быть проблемой в будущем, то лучше это проверить сейчас на исторических данных
3. Если вы проектируете вашу систему держа в голове инструментарий метрик, то ваши дела пойдут в гору в будущем. В частности, вы не найдете себя грепающим логи, чтобы измерить ваши показатели.
4. Вы заметите, какие вещи меняются и что остается неизменным. Например, предположим, вы хотите напрямую оптимизировать “однодневных” пользователей. Однако во время ваших прошлых манипуляций вы можете заметить, что сильные изменения в пользовательском опыте не влияют на метрику.

Команда Google Plus измеряет “открытий на просмотр”, “репостов на просмотр”, плюсование(лайк или +1) на просмотр, отношение комментариев к просмотрам, комментарии на пользователя, репосты на пользователя и другие что можно использовать в расчете добротности сообщения во время обслуживания. Также, обратите внимание на экспериментальный фреймворк, который вы можете сгруппировать пользователей в кластеры и агрегироваться статистику для экспериментов, что важно. Смотрите правило **#12** .

Будьте более свободны(не ограничивайте себя) в сборе метрик, таким образом вы получите более широкое представление о вашей системе. 
Заметили проблему? Добавьте метрику и отслеживайте её! Волнуетесь о некоторых количественных изменениях после последнего релиза? Добавьте метрику и следите за этим! 


#### Правило #3: Используйте методы машинного обучения вместо сложной эвристики

Простая эвристика может открыть двери вашему продукту. Сложные эвристики трудно поддерживаемы. Как только у вас есть данные и представление о том чего вы пытаетесь достигнуть, переходите к машинному обучению. Как и в большинстве инженерных задачах программного обеспечения, вы захотите постоянно обновлять свой подход, будь то эвристики или ML-модели и вы найдете что ML-модели проще в обновлении и обслуживании(см. правило **#16**)


## Первый пайплайн (Pipeline)

Сосредоточьтесь на своей системной инфраструктуре для вашего первого пайплайна. Хотя интересно думать и воображать про всякие штуки, которое вы собираетесь делать с помощью машинного обучения, но вам будет сложно понять, что происходит, если вы не доверяете своему пайплайну.

#### Правило #4: Сохраните первую модель простой и получите правильную инфраструктуру.
 
Первая модель обеспечивает большой толчок вашему продукту, и это не должно быть фантазией. Но вы столкнетесь с куда большим числом инфраструктурных задач, чем ожидаете. До того как кто-то сможет использовать вашу новую систему машинного обучения, вы должны определить:
1. Как и где получить обучающую выборку
2. В первом приближении определить что такое хорошо и плохо для вашей системы
3. Как внедрить модель в ваше приложение. Вы можете применить модель в реальном времени или предрасчитать модель на примерах в оффлайне и сохранить результаты в таблице. Например, вы желаете классифицировать веб-страницы и сохранить результаты в в таблицу, но вы можете хотеть классифицировать сообщения чата в реалтайме.

Выбор простых признаков делает это проще  и убедитесь что:
1. Функции(Признаки) правильно реализуют ваш алгоритм обучения
2. Модель обучается с разумными весами(избегайте переобучения)
3. Признаки правильно передаются в вашу модель на сервере

Если ваша система делает три этих вещи надежно, то вы выполнили большую часть работы. Ваша простая модель дает вам базовые показатели и базовое поведение, которое вы можете использовать для тестирования более сложных моделей. Некоторые команды стремятся к нейтральному первому запуску: первый запуск явным образом не ставит целью получение прибыли через машинное обучение, чтобы не отвлекаться.

#### Правило #5: Проверяйте инфраструктуру независимо от машинного обучения.

Убедитесь, что инфраструктура тестируема и части обучающей системы инкапсулированы так, что вы можете протестировать каждую деталь. В частности:

1. Проверьте получение данных внутри алгоритма. Проверяйте что признаки которые должны быть заполнены действительно заполнены. Если, вы не нарушаете конфиденциальность, то проверьте вручную  входные данные в свой алгоритм обучения. Если возможно, проверьте статистики в вашем папйплайне, по сравнению с другими данными, например с помощью RASTA.
2. Проверьте получение моделей из обучающего алгоритма. Убедитесь, что модель на обучении дает те же результаты, что и модель в рабочей среде. (см. правило **#37**)

У машинного обучения есть элемент непредсказуемости, так что убедитесь что имеете тесты для кода создающие примеры для обучающей и реальной среде, и что вы можете загружать и использовать фиксированную модель в реальной среде. Кроме того, важно понимать ваши данные: см. [Практические советы по анализу больших, сложных наборов данных (ENG)](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html).

#### Правило #6: Будьте осторожны с удалением данных когда копируете пайплайн.

Часто мы создаем пайплайн копируя существующий пайплайн (см. [культ карго programming](https://en.wikipedia.org/wiki/Cargo_cult_programming)), а старый пайплайн удаляет данные которые нужны для нового пайплайну. Например, пайплайн для Google Plus горяче удалял старые сообщения(потому что пытается ранжировать новые сообщения). Этот пайплайн скопирован и использован в Google Plus Stream, где старые сообщения важны, но пайплайн удалял старые сообщения. Другой распространенный шаблон состоит в том что только логгировать данные которые пользователь смотрят. Таким образом, эти данные бесполезны, если мы хотим моделировать почему конкретное сообщение не было замечено пользователем, потому что все негативные примеры были удалены. Похожий случай произошел в Play. Во время работы над “Play Apps Home”, был создан новый пайплайн, который также содержал примеры двух других лендингов (“Play Games Home” and “Play Home Home”) без какого-то признака позволяющего понять откуда пришел каждый объект.

#### Правило #7: Включите эвристики в признаки или обработайте их

Обычно задачи, которые решают с помощью машинного обучения, не являются новыми. Существуют системы для ранжирования , классификации или любой другой задачи, которую вы пытаетесь решить. Это означает что существует множество правил и эвристик. Эти эвристики могут дать вам прирост при настройке модели машинного обучения. Ваши эвристики должны быть извлечены для любой информации и на это есть пара причин. Во-первых, это обеспечит плавный переход на использование машинного обучения. Во-вторых, обычно такие правила включают в себя интуитивные знания о системе и мы не хотели бы их потерять
Вот 4 подхода как вы можете использовать существующие эвристики:

1. Предобработка с использованием эвристики. Если признак невероятно хорош, то это вариант. Например, в спам фильтре, если отправитель уже находится в черном списке, то не нужно переучивать и менять значение “черный список”. Заблокируйте сообщение. Этот подход имеет наибольший смысл в задачах бинарной классификации.
2. Создавайте новые признаки. Создание признаков из эвристик это превосходно. К пример, если вы используете эвристику чтобы рассчитать оценку релевантности результатов запроса, вы можете включать оценку значения этого признака. Позже вы можете использовать техники машинного обучения для получения правильного значения(к примеру, преобразование значения в один из конечных  наборов дискретных значений или объединение с другими признаками), но начинайте с исходных значений произведенными на основе эвристик. 
3. Добавляйте сырые данные из эвристик. Если существует эвристика для приложений, которая объединяет количество установок, количество символов в тексте и день недели, то подумайте о том, чтобы разделить эти части друг от друга и передать в обучение отдельно. Здесь применяются некоторые методы, применимые к ансамблям (см. **Правило #40**)
4. Изменяйте метку. Этот вариант применяется , если вы чувствуете, что эвристика содержит информацию не отраженную в метке. К примеру, если вы пытаетесь максимизировать количество загрузок, но также вы нуждаетесь в качественном контенте, то возможно решение заключается в умножении метки на средний рейтиг приложения. Здесь много пространства для маневра. Смотрите раздел “Ваша первая цель”.
Обратите внимание, что вы добавляете сложности, когда используете эвристики в ML-системе. Использование старых эвристик в новом ML-алгоритме позволяет осуществить плавный переход, но подумайте о том, что может есть просто путь достигнуть того же эффекта.

### Мониторинг

> В общем, правилами хорошего тона считаются, создание уведомлений на действия и наличие дашбордов.

#### Правило #8: Знайте требования к актуальности вашей системы.

Насколько ухудшается производительность, если ваша модель устареет на 1 день? А на неделю? А на квартал? Это информация поможет вам понять приоритеты вашего мониторинга. Если вы потеряете 10% своего дохода из-за того что модель не обновляется в течении дня, имеет смысл постоянно следить за её работой. В большинстве рекламных систем новые объявления обрабатываются каждый день и ежедневно обновляются. Например, если ML-модель в Google Play Search не будет обновлена, это может повлиять на месячный доход. Некоторые модели для What’s Hot in Google Plus не имеют идентификаторов сообщений в своей модели, поэтому они могут экспортировать этим модели нечасто. Другие модели, у которых есть идентификаторы сообщения, обновляются гораздо чаще. Также обратите внимание, что актуальность может меняться со временем, особенно когда признаки добавляются или удаляются из вашей модели.

#### Правило #9: Обнаружьте проблемы до экспорта модели.

Многие МЛ-системы имеют этап, на котором вы экспортируете модель в продакшн. Если проблема связана с экспортируемой моделью, то будут проблемы с пользовательским интерфейсом. А если проблема будет обнаружена заранее - на обучении, то пользователи ничего не заметят.

Выполняйте проверку работоспособности, прямо перед экспортом модели. В частности, проверьте  эффективность(качество) модели на отложенных данных. Или, если у вас есть длительная проблема с данными, то не экспортируйте модель. Многие команды, постоянно развертывающие модели, проверяют площадь под [ROC-кривой](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) ((или [AUC](http://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it)) перед экспорта.  **Проблемы с моделями, которые не были экспортированы требуют уведомления по электронной почте, но для проблем с моделями, которые задействованы в пользовательском интерфейсе, может потребоваться отдельная страница.** Поэтому лучше подождать и быть уверенным, прежде чем воздействовать на пользователей.


#### Правило #10: Следите за тихими неудачами
 
Это проблема возникает чаще для МЛ-система, чем для других видов систем. Предположим, что конкретная, таблица, которую вы соединяете больше не обновляется. МЛ-система будет корректироваться и поведение будет по прежнему хорошим в течении долгого времени, однако постепенно деградируя. Иногда поиск таблиц, которые давно устарели и простое обновление улучшает эффективность больше, чем любой другой запуск на в квартале. К примеру, покрытие признака может измениться из-за изменения в реализации: к примеру признак-столбец может быть заполнен на 90% и внезапно упал до 60% объектов. В сервисе Play была таблица, которая не обновлялась 6 месяцев, и обновление таблицы привело к увеличению на 2% количества установок. Если вы отслеживаете статистики в данных, а также вручную проверяете аномалии в данных, то вы можете уменьшить такие сбои. *

* <sup> [*A Framework for Analysis of Data Freshness* - Bouzeghoub & Peralta](https://www.fing.edu.uy/inco/grupos/csi/esp/Publicaciones/2004/iqis2004-mb.pdf)</sup>

#### Правило #11: Назначьте владельцев наборам признаков и документации. 

Если система большая и имеет много признаков, вам нужно знать кто создал и поддерживал каждый признак. Если вы обнаружите, что человек, который понимает признак , уходит, то убедитесь что он передаст информацию кому-то. Хотя, многие признаки имеют понятные имена, полезно иметь более подробное описание, что значит каждый признак, откуда он взялся, и как он может помочь(вносит вклад).


### Ваша первая цель

> У вас есть много показателей или измерений относительно системы, о которой вы беспокоитесь, но для МЛ-алгоритма часто требуется одна **цель - это число, которое ваш алгоритм “пытается” оптимизировать.** Здесь я различаю цели и показатели: **метрика - это любое число, которая сообщает ваша система**, и она может быть важной, а может быть нет. Смотрите правило **#2**.


#### Правило #12: Не переусердствуйте, в выборе цели которую вы напрямую оптимизируйте.

Вы хотите заработать деньги, сделать ваших пользователей счастливыми и сделать мир лучше. Есть масса метрик, которые вас волнуют, и вы должны их измерить (см. Правило **#2**). Однако, в начале процесса машинного обучение, вы заметите, что все они растут, даже те, которые вы не оптимизируете напрямую. Например, предположим что вы беспокоитесь о количестве кликов, времени пребывания на сайте, и количеством пользователей за день. Если вы оптимизируете количество кликов, вы скорее всего, увидите как время пребывания на сайте увеличивается.

Итак, сохраняйте его простым и не задумывайтесь о сложном балансе различных метрик, когда вы еще можете легко увеличить все показатели. Не используйте это правило повсеместно: не путайте свою цель с работоспособностью  системы (см правило 39). И, **если вы обнаружите  что увеличиваете напрямую оптимизируемую метрику, но решите не запускать, может потребоваться какая-то объективная ревизия.** 


#### Выбирайте простую, поддающуюся наблюдениям и понятную метрику для вашей первой цели.

Часто вы не знаете, какова истинная цель. Вы думаете, что делаете, а затем, когда вы смотрите на данные и анализируете бок-о-бок вашу старую систему и новую систему МЛ, вы понимаете, что хотите её настроить. Кроме того разные члены команды часто не могут договориться об истинной цели. **Цель МЛ должна быть легкой для измерения и прозрачной для “настоящей” цели.** Поэтому обучайтесь  простой МЛ-цели и подумайте о том, что на самом деле  есть “слой политики”, который позволяет вам добавить дополнительную логику (надеюсь, очень простую логику), чтобы сделать окончательный рейтинг. 

Простейшей моделью является поведение пользователя, которое непосредственно наблюдается и относится к действию системы:

1. Была ли нажата эта ранжированная ссылка?
2. Был ли загружен этот ранжированный объект?
3. Был ли этот ранжированный объект перенаправлен/был ли ответ на него  или он отправлен по email?
4. Был ли оценен этот ранжированный объект?
5. Был ли показанный объект отмечен как спам/порнография/оскорбления?

Поначалу избегайте моделирования косвенных эффектов:

1. Вернулся ли посетитель на следующий день?
2. Как долго пользователь был на сайте?
3. Сколько пользователей ежедневно активны?

Косвенные эффекты хорошо улучшают показатели и могут быть использованы в А/Б тестировании и запуске решений. 
В заключении, не пытайтесь получить от МЛ ответ:
1. Счастлив ли пользователь использующий ваш продукт?
2. Доволен ли пользователь взаимодействием?
3. Улучшает ли продукт общее благополучие пользователя?
4. Как это влияет на общее состояние компании?

Все это очень важно, и в тоже время невероятно сложно. Вместо этого, используйте “допущение-следствие”: если пользователи счастливы, то они останутся дольше на сайте. Если пользователь доволен взаимодействием, значит он придет и завтра. Что касается благополучия и здоровья компании, человеческое суждение требуется для того, чтобы связать любую цель МЛ  с характером продукта, который вы продаете и вашим бизнес планом, поэтому мы не закончим [здесь](https://www.youtube.com/watch?v=bq2_wSsDwkQ).

#### Правило #14: Начиная с интерпретируемой модели вы упрощаете отладку.

[Линейная регрессия](https://en.wikipedia.org/wiki/Linear_regression), [логистическая регрессия](https://en.wikipedia.org/wiki/Logistic_regression), и [Пуассоновская регрессия](https://en.wikipedia.org/wiki/Poisson_regression) напрямую определяются как вероятностная модель. Каждое предсказание интерпретируется как вероятность или ожидаемое значение. Это облегчает их отладку, чем модели, которые используют цели (zero­one loss, various hinge losses, et cetera), которые пытаются напрямую оптимизировать  точность классификации или эффективность ранжирования. К примеру, если вероятности при обучении отклоняются от вероятностей,  предсказанных бок-о-бок или путем проверки рабочей системы, это отклонение может выявить проблему.

Например, в линейной, логистической или регрессии Пуассона **имеются подмножества данных, где среднее прогнозируемое ожидание равно средней метке (1-й момент калиброван или просто откалиброван)<sup>3</sup>**. Если у вас есть признак, который равен 1 или 0 для каждого примера, тогда набор примеров, где этот признак равен 1, откалиброван. Кроме того, если у вас есть признак, который равен 1 для каждого примера, тогда набор всех примеров откалиброван.
 
С простыми моделями проще получать обратную связь (см правило 36).
Часто, мы используем эти вероятностные модели для принятия решений: таких как ранжирование сообщений в убывающем порядке по ожидаемому значению( т.е. вероятности от клика/загрузки/ чего то другого). **Однако, помните когда придет время выбрать какую модель использовать, решение имеет большее значение, чем вероятность получения данных данной модели** (см правило № 27). 

<sup> (3) Это верно, если у вас нет регуляризации и ваш алгоритм сходится. В целом это примерно так.</sup>

#### Правило #15: Разделяйте фильтрацию спама и ранжирование качества в слое политик.

Ранжирование качества это изобразительное искусство, но фильтрация спама это война*. Сигналы, которые вы используете для определения высококачественных сообщений, станут очевидными тем кто использует вашу систему и они будут настраивать свои сообщения, чтобы достигнуть этих свойств. Таким образом, ваш рейтинг качества должен быть сосредоточен на ранжировании контента, который публикуется с хорошим умыслом. Вы не должны делать скидку при обучении ранжированию спаму. **Аналогичным образом,  “неприйстойный” контент должен обрабатываться отдельно от качественного ранжирования.** Фильтрация спама - это отдельная история. Вы должны быть готовы, что признаки которые вам нужно создать, будут постоянно меняться. Часто будут введены очевидные правила, которые вы запрограммируете в системе (если сообщение имеет более трех спам-голосов, не извлекает их и т. д.). Любая научная модель должна обновляться ежедневно, если не быстрее. Репутация создателя контента будет играть большую роль.

На некотором уровне, выход этих двух систему может быть объединен. Имейте ввиду, что, фильтрация спама в результатах поиска должна быть более агрессивной чем фильтрация спама в сообщениях электронной почты. Кроме того, стандартной практикой является удаление спама из обучающих данных для  качественного классификатора.

<sup>[Google Research Blog - Lessons learned while protecting Gmail](https://research.googleblog.com/2016/03/lessons-learned-while-protecting-gmail.html?m=1)</sup>

## ML этап II: Feature Engineering

> На первом этапе жизненого цикла системы машинного обучения важными задачами являются получение данных для обучения в обучающую систему, получение любых показателей, представляющих интерес и создание обслуживающей инфраструктуры. **После того, как у вас есть работающая "от и до" система с модульными и системными тестами, начинается этап 2.**

На втором этапе есть много неприятных особенностей. Существует множество очевидных признаков, которые можно добавить в систему. Таким образом, вторая фаза машинного обучения включает в себя использование как можно большего числа признаков и объединение их интуитивно понятными способами. На этом этапе все показатели должны расти. Будет много запусков, и это прекрасное время чтобы подключить достаточно много инженеров, которые могут объединить все данные, необходимые для создания действительно классной системы обучения.

#### Правило 16 - Планируйте запуск и итерации.

Не ожидайте, что модель, над которой вы сейчас работаете, будет последней, которую вы запустите, и даже тогда когда вы перестанете запускать модели. Подумайте, будет ли сложность, которую вы добавляете с этим запуском, замедлять будущие запуски. Многие команды запустили модель через квартал или более в течение многих лет. Существуют три основные причины запуска новых моделей:

1. Вы придумываете новые признаки,
2. Вы настраиваете регуляризацию и комбинируете старые признаки по-новому
3. И/или вы корректируете цель.

Несмотря на все это, хорошо бы все равно дать вашей модели немного: изучение данных, используемых как пример, может помочь найти новые сигналы, а также старые(сломанные). Поэтому, когда вы строите свою модель, подумайте о том, как легко добавлять или удалять или рекомбинировать признаки. Подумайте, как легко создать новую копию конвейера и проверить его правильность. Подумайте, возможно ли иметь два или три экземпляра, работающих параллельно. Наконец, не беспокойтесь о том, входит ли признак 16 из 35 в эту версию конвейера. Вы добавите его в следующем квартале.

#### Правило 17 - Начните с непосредственно наблюдаемых и задокументированных признаков вместо "обученных" признаков.

Это может быть спорным моментом, но это позволяет избежать множества подводных камней. Прежде всего, дадим определение, что такое "обученный" признак. "Обученный" признак - это признак, созданный либо внешней системой (например, в ходе кластеризации), либо в процессе обучения (например, через факторизационную модель или глубокое обучение). Оба эти способа могут быть полезны, но у них может быть много проблем, поэтому они не должны быть в первой модели.

Если вы используете внешнюю систему для создания признака, помните, что система имеет свою собственную цель. Цель внешней системы может быть слабо коррелирована с вашей текущей задачей. Если вы получите слепок(snapshot) внешней системы, то он может устареть. Если вы обновите признаки из внешней системы, значения могут измениться. Если вы все таки используете внешнюю систему для генерации(наполнения) признака, то имейте в виду, что требуется большая осторожность.

Основная проблема с факторизационными моделями и глубоким оубчением заключается в том, что они "не выпуклые". Таким образом, нет никакой гарантии, что оптимальное решение может быть аппроксимировано или найдено, а локальные минимумы, найденные на каждой итерации, могут быть разными. Этот вариант затрудняет оценку изменения на вашу систему: является оно значимым или случайным. Создавая модель без глубоких признаков, вы можете получить отличную базовую эффективность. После достижения этой базовой линии(baseline) вы можете попробовать более сложные(эзотерические) подходы.

#### Rule 18 - Explore with features of content that generalize across contexts.

#### Правило 18 - Исследуйте особенности контента, которые обобщают контексты.

Часто система машинного обучения это только малая часть одной большео картины. 
Например, если вы представите пост, который может быть использован в разделе "Что нового"(What’s Hot), множество людей лайкнут, отрепостят или прокомментируют пост даже прежде, чем он будет показан в разделе "Что нового"(What’s Hot). Если вы предоставляете такую статистику алгоритму обучения, это может продвинуть новые посты, которые не имеют данных, в том контексте который оптимизируется. YouTube Watch Next мог использовать количество просмотров, или со-просмотры (количество просмотров одного видео после другого) из поиска YouTube. Вы также можете использовать явные пользовательские рейтинги.

Наконец, если у вас есть действие пользователя, которое вы используете в качестве метки, наблюдение такого же действия над документом в другом контексте может стать отличным признаком. Все эти признаки позволяют вам вводить новый контент в контекст. 

Обратите внимание, что речь идет не о персонализации: выясните, нравится ли кому-то сначала контент в этом контексте, а затем выясните, кому это нравится больше или меньше.

#### Правило 19 - Используйте очень спецефические признаки, когда сможете.

С огромным количеством данных проще получить миллион простых признаков, чем несколько сложных признаков. Идентификаторы получаемых документов и канонизированные запросы не дают большого обобщения, но выравнивают ваш рейтинг с вашими метками на главных запросах. Таким образом, не бойтесь групп признакой, где каждый признак применяется к очень небольшой части ваших данных, но общий охват превышает 90%. Вы можете использовать регуляризацию для устранения признаков, которые используются лишь в нескольких примерах.

#### Rule 20 - Combine and modify existing features to create new features in human-understandable ways.

There are a variety of ways to combine and modify features. Machine learning systems such as TensorFlow allow you to pre­process your data through [transformations](https://www.tensorflow.org/tutorials/linear/overview#feature-columns-and-transformations). The two most standard approaches are “discretizations” and “crosses”.

Discretization consists of taking a continuous feature and creating many discrete features from it. Consider a continuous feature such as age. You can create a feature which is 1 when age is less than 18, another feature which is 1 when age is between 18 and 35, et cetera. Don’t overthink the boundaries of these histograms: basic quantiles will give you most of the impact. Crosses combine two or more feature columns. A feature column, in TensorFlow's terminology, is a set of homogenous features, (e.g. {male, female}, {US, Canada, Mexico}, et cetera). A cross is a new feature column with features in, for example, *{male, female} × {US,Canada, Mexico}*. This new feature column will contain the feature (male, Canada). If you are using TensorFlow and you tell TensorFlow to create this cross for you, this (male, Canada) feature will be present
in examples representing male Canadians. Note that it takes massive amounts of data to learn models with crosses of three, four, or more base feature columns.

Crosses that produce very large feature columns may overfit. For instance, imagine that you are doing some sort of search, and you have a feature column with words in the query, and you
have a feature column with words in the document. You can combine these with a cross, but you will end up with a lot of features (see Rule **#21**). When working with text there are two
alternatives. The most draconian is a dot product. A dot product in its simplest form simply counts the number of common words between the query and the document. This feature can
then be discretized. Another approach is an intersection: thus, we will have a feature which is present if and only if the word “pony” is in the document and the query, and another feature
which is present if and only if the word “the” is in the document and the query.

#### Rule 21 - The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.

There are fascinating statistical learning theory results concerning the appropriate level of complexity for a model, but this rule is basically all you need to know. I have had conversations in which people were doubtful that anything can be learned from one thousand examples, or that you would ever need more than 1 million examples, because they get stuck in a certain method of learning. The key is to scale your learning to the size of your data:

1. If you are working on a search ranking system, and there are millions of different words in the documents and the query and you have 1000 labeled examples, then you should use a dot product between document and query features, [TF­IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), and a half-­dozen other highly human-­engineered features. 1000 examples, a dozen features.
2. If you have a million examples, then intersect the document and query feature columns, using regularization and possibly feature selection. This will give you millions of features,
but with regularization you will have fewer. Ten million examples, maybe a hundred thousand features.
3. If you have billions or hundreds of billions of examples, you can cross the feature columns with document and query tokens, using feature selection and regularization. You will have a billion examples, and 10 million features.

Statistical learning theory rarely gives tight bounds, but gives great guidance for a starting point.
In the end, use Rule **#28** to decide what features to use.

#### Rule 22 - Clean up features you are no longer using.

Unused features create technical debt. If you find that you are not using a feature, and that combining it with other features is not working, then drop it out of your infrastructure. You want to keep your infrastructure clean so that the most promising features can be tried as fast as
possible. If necessary, someone can always add back your feature. Keep coverage in mind when considering what features to add or keep. How many examples are covered by the feature? For example, if you have some personalization features, but only
8% of your users have any personalization features, it is not going to be very effective. At the same time, some features may punch above their weight. For example, if you have a
feature which covers only 1% of the data, but 90% of the examples that have the feature are positive, then it will be a great feature to add.

### Human Analysis of the System

> Before going on to the third phase of machine learning, it is important to focus on something that is not taught in any machine learning class: how to look at an existing model, and improve it. This is more of an art than a science, and yet there are several anti-­patterns that it helps to avoid.

#### Rule 23 - You are not a typical end user.*

This is perhaps the easiest way for a team to get bogged down. While there are a lot of benefits to fish-fooding (using a prototype within your team) and dog-fooding (using a prototype within your company), employees should look at whether the performance is correct. While a change which is obviously bad should not be used, anything that looks reasonably near production should be tested further, either by paying laypeople to answer questions on a crowdsourcing platform, or through a live experiment on real users. There are two reasons for this. The first is that you are too close to the code. You may be looking for a particular aspect of the posts, or you are simply too emotionally involved (e.g. confirmation bias). The second is that your time is too valuable. Consider the cost of 9 engineers sitting in a one hour meeting, and think of how many contracted human labels that buys on a crowdsourcing platform.

If you really want to have user feedback, **use user experience methodologies**. Create user personas (one description is in Bill Buxton’s [~~Designing~~ *Sketching User Experiences*](https://www.amazon.com/Sketching-User-Experiences-Interactive-Technologies/dp/0123740371)) early in a process and
do usability testing (one description is in Steve Krug’s [*Don’t Make Me Think*](https://www.amazon.com/Dont-Make-Me-Think-Usability/dp/0321344758)) later. User personas involve creating a hypothetical user. For instance, if your team is all male, it might help to design a 35­-year old female user persona (complete with user features), and look at the results it generates rather than 10 results for 25­-40 year old males. Bringing in actual people to watch their reaction to your site (locally or remotely) in usability testing can also get you a fresh perspective.

<sup>[Google Research Blog - How to measure translation quality in your user interfaces](https://research.googleblog.com/2015/10/how-to-measure-translation-quality-in.html?m=1)

#### Rule 24 - Measure the delta between modules

One of the easiest, and sometimes most useful measurements you can make before any users have looked at your new model is to calculate just how different the new results are from production. For instance, if you have a ranking problem, run both models on a sample of queries through the entire system, and look at the size of the symmetric difference of the results
(weighted by ranking position). If the difference is very small, then you can tell without running an experiment that there will be little change. If the difference is very large, then you want to make sure that the change is good. Looking over queries where the symmetric difference is high
can help you to understand qualitatively what the change was like. Make sure, however, that the system is stable. Make sure that a model when compared with itself has a low (ideally zero)
symmetric difference.

#### Rule 25 - When choosing models, utilitarian performance trumps predictive power.

Your model may try to predict click­-through-­rate. However, in the end, the key question is what you do with that prediction. If you are using it to rank documents, then the quality of the final ranking matters more than the prediction itself. If you predict the probability that a document is spam and then have a cutoff on what is blocked, then the precision of what is allowed through matters more. Most of the time, these two things should be in agreement: when they do not
agree, it will likely be on a small gain. Thus, if there is some change that improves log loss but degrades the performance of the system, look for another feature. When this starts happening more often, it is time to revisit the objective of your model.

#### Rule 26 - Look for patterns in the measured errors, and create new features.

Suppose that you see a training example that the model got “wrong”. In a classification task, this could be a false positive or a false negative. In a ranking task, it could be a pair where a positive was ranked lower than a negative. The most important point is that this is an example that the
machine learning system knows it got wrong and would like to fix if given the opportunity. If you give the model a feature that allows it to fix the error, the model will try to use it.
On the other hand, if you try to create a feature based upon examples the system doesn’t see as mistakes, the feature will be ignored. For instance, suppose that in Play Apps Search,
someone searches for “free games”. Suppose one of the top results is a less relevant gag app. So you create a feature for “gag apps”. However, if you are maximizing number of installs, and people install a gag app when they search for free games, the “gag apps” feature won’t have the effect you want.

Once you have examples that the model got wrong, look for trends that are outside your current feature set. For instance, if the system seems to be demoting longer posts, then add post
length. Don’t be too specific about the features you add. If you are going to add post length, don’t try to guess what long means, just add a dozen features and the let model figure out what to do with them (see Rule **#21**). That is the easiest way to get what you want.

#### Rule 27 - Try to quantify observed undesirable behavior.

Some members of your team will start to be frustrated with properties of the system they don’t like which aren’t captured by the existing loss function. At this point, they should do whatever it takes to turn their gripes into solid numbers. For example, if they think that too many “gag apps” are being shown in Play Search, they could have human raters identify gag apps. (You can feasibly use human-­labelled data in this case because a relatively small fraction of the queries account for a large fraction of the traffic.) If your issues are measurable, then you can start using them as features, objectives, or metrics. The general rule is **“measure first, optimize second”**.

#### Rule 28 - Be aware that identical short-term behavior does not imply identical long-term behavior.

Imagine that you have a new system that looks at every doc_id and exact_query, and then calculates the probability of click for every doc for every query. You find that its behavior is
nearly identical to your current system in both side by sides and A/B testing, so given its simplicity, you launch it. However, you notice that no new apps are being shown. Why? Well,
since your system only shows a doc based on its own history with that query, there is no way to learn that a new doc should be shown.

The only way to understand how such a system would work long­term is to have it train only on data acquired when the model was live. This is very difficult.

### Training-Serving Skew

> Training­-serving skew is a difference between performance during training and performance
during serving. This skew can be caused by:
* a discrepancy between how you handle data in the training and serving pipelines, or
* a change in the data between when you train and when you serve, or
* a feedback loop between your model and your algorithm.

> We have observed production machine learning systems at Google with training-­serving skew
that negatively impacts performance. The best solution is to explicitly monitor it so that system
and data changes don’t introduce skew unnoticed.

#### Rule 29 - The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time.

Even if you can’t do this for every example, do it for a small fraction, such that you can verify the consistency between serving and training (see Rule **#37**). Teams that have made this measurement at Google were sometimes surprised by the results. YouTube home page switched to logging features at serving time with significant quality improvements and a reduction in code complexity, and many teams are switching their infrastructure as we speak.

#### Rule 30 - Importance weight sampled data, don't arbitrarily drop it!

When you have too much data, there is a temptation to take files 1­12, and ignore files 13­99. This is a mistake: dropping data in training has caused issues in the past for several teams (see Rule **6**). Although data that was never shown to the user can be dropped, importance weighting is best for the rest. Importance weighting means that if you decide that you are going
to sample example X with a 30% probability, then give it a weight of 10/3. With importance weighting, all of the calibration properties discussed in Rule **#14** still hold.

#### Rule 31 - Beware that if you join data from a table at training and serving time, the data in the table may change.

Say you join doc ids with a table containing features for those docs (such as number of comments or clicks). Between training and serving time, features in the table may be changed.
Your model's prediction for the same document may then differ between training and serving. The easiest way to avoid this sort of problem is to log features at serving time (see Rule **#32**). If the table is changing only slowly, you can also snapshot the table hourly or daily to get reasonably close data. Note that this still doesn’t completely resolve the issue.

#### Rule 32 - Re-use code between your training pipeline and your serving pipeline whenever possible.

Batch processing is different than online processing. In online processing, you must handle each request as it arrives (e.g. you must do a separate lookup for each query), whereas in batch
processing, you can combine tasks (e.g. making a join). At serving time, you are doing online processing, whereas training is a batch processing task. However, there are some things that
you can do to re­use code. For example, you can create an object that is particular to your system where the result of any queries or joins can be stored in a very human readable way,
and errors can be tested easily. Then, once you have gathered all the information, during serving or training, you run a common method to bridge between the human-­readable object
that is specific to your system, and whatever format the machine learning system expects. **This eliminates a source of training-­serving skew.** As a corollary, try not to use two different programming languages between training and serving ­ that decision will make it nearly impossible for you to share code.

#### Rule 33 - If you produce a model based on the data until January 5th, test the model on the data from January 6th and after.

In general, measure performance of a model on the data gathered after the data you trained the model on, as this better reflects what your system will do in production. If you produce a model based on the data until January 5th, test the model on the data from January 6th. You will expect that the performance will not be as good on the new data, but it shouldn’t be radically worse. Since there might be daily effects, you might not predict the average click rate or conversion rate, but the area under the curve, which represents the likelihood of giving the positive example a score higher than a negative example, should be reasonably close.

#### Правило 34 - В бинарной классификации для фильтрации (таких как детекция спама или определении интересных писем), делайте небольшие краткосрочные жертвыв качестве для получения более чистых данных.

В задаче фильтрации спама, объекты отмеченные как спам не показываются пользователю. Предположим, у вас есть фильтр, который блокирует 75% всех нежелательных писем. Вы можете попробовать собрать дополнительные обучающие данные из писем, которые показываете пользователю. Например, если пользователь  отметил письмо как спам прошедший через ваш фильтр, то вы можете узнать об этом.

Но этот подход приходит к смещению выборки<sup>*</sup>. Вы можете собирать более чистые данные, если во время работы 1% всех ваших меток будет откладываться. Сейчас ваш фильтр блокируется около 74% негативных примеров. Эта отложенная выборка и есть  пример как получить обучающие данные. 

Обратите внимание, что если вы фильтруете 95% негативных примеров или больше, этот подход менее жизнеспособный. Тем не менее, если вы хотите измерить эффективность работы, вы можете сделать меньше откладывать образцов (например 0.1% или 0.001%). Десять тысяч образцов достаточно, чтобы оценить эффективность довольно точно.

<sup>*</sup> [Про Bias можно прочитать здесь](https://codesachin.wordpress.com/2015/08/05/on-the-biasvariance-tradeoff-in-machine-learning/)


#### Rule 35 - Beware of the inherent skew in ranking problems.

When you switch your ranking algorithm radically enough that different results show up, you have effectively changed the data that your algorithm is going to see in the future. This kind of skew will show up, and you should design your model around it. There are multiple different approaches. These approaches are all ways to favor data that your model has already seen.

1. Have higher regularization on features that cover more queries as opposed to those features that are on for only one query. This way, the model will favor features that are
specific to one or a few queries over features that generalize to all queries. This approach can help prevent very popular results from leaking into irrelevant queries. Note
that this is opposite the more conventional advice of having more regularization on feature columns with more unique values.
2. Only allow features to have positive weights. Thus, any good feature will be better than a feature that is “unknown”.
3. Don’t have document­only features. This is an extreme version of #1. For example, even if a given app is a popular download regardless of what the query was, you don’t want to
show it everywhere<sup>4</sup>. Not having document­only features keeps that simple.

<sup>4 - The reason you don’t want to show a specific popular app everywhere has to do with the importance of
making all the desired apps reachable. For instance, if someone searches for “bird watching app”, they
might download “angry birds”, but that certainly wasn’t their intent. Showing such an app might improve
download rate, but leave the user’s needs ultimately unsatisfied.</sup>

#### Rule 36 - Avoid feedback loops with positional features.

The position of content dramatically affects how likely the user is to interact with it. If you put an app in the first position it will be clicked more often, and you will be convinced it is more likely to be clicked. One way to deal with this is to add positional features, i.e. features about the position of the content in the page. You train your model with positional features, and it learns to weight, for example, the feature "1st­position" heavily. Your model thus gives less weight to other factors for examples with "1st­position=true". Then at serving you don't give any instances the positional feature, or you give them all the same default feature, because you are scoring candidates before you have decided the order in which to display them. Note that it is important to keep any positional features somewhat separate from the rest of the
model because of this asymmetry between training and testing. Having the model be the sum of a function of the positional features and a function of the rest of the features is ideal. For example, don’t cross the positional features with any document feature.

#### Rule 37 - Measure training/serving skew.

There are several things that can cause skew in the most general sense. Moreover, you can divide it into several parts:

1. The difference between the performance on the training data and the holdout data. In general, this will always exist, and it is not always bad.
2. The difference between the performance on the holdout data and the “next­day” data. Again, this will always exist. **You should tune your regularization to maximize the next­day performance.** However, large drops in performance between holdout and next­day data may indicate that some features are time-­sensitive and possibly degrading
model performance.
3. The difference between the performance on the “next­day” data and the live data. If you apply a model to an example in the training data and the same example at serving, it
should give you exactly the same result (see Rule **#5**). Thus, a discrepancy here probably indicates an engineering error.

## Slow Growth and Optimization and Complex models

> There will be certain indications that the second phase is reaching a close. First of all, your monthly gains will start to diminish. You will start to have tradeoffs between metrics: you will see some rise and others fall in some experiments. This is where it gets interesting. Since the gains
are harder to achieve, the machine learning has to get more sophisticated. A caveat: this section has more blue-­sky rules than earlier sections. We have seen many teams
go through the happy times of Phase I and Phase II machine learning. Once Phase III has been reached, teams have to find their own path.

#### Rule 38 - Don't waste time on new features if unaligned objectives have become the issue.

As your measurements plateau, your team will start to look at issues that are outside the scope of the objectives of your current machine learning system. As stated before, if the product goals are not covered by the existing algorithmic objective, you need to change either your objective
or your product goals. For instance, you may optimize clicks, plus-­ones, or downloads, but make launch decisions based in part on human raters.

#### Rule 39 - Launch decisions are a proxy for long-term product goals.

Alice has an idea about reducing the logistic loss of predicting installs. She adds a feature. The
logistic loss drops. When she does a live experiment, she sees the install rate increase. However, when she goes to a launch review meeting, someone points out that the number of
daily active users drops by 5%. The team decides not to launch the model. Alice is disappointed, but now realizes that launch decisions depend on multiple criteria, only some of
which can be directly optimized using ML. The truth is that the real world is not dungeons and dragons: there are no “hit points” identifying the health of your product. The team has to use the statistics it gathers to try to effectively
predict how good the system will be in the future. They need to care about engagement, 1 day active users (DAU), 30 DAU, revenue, and advertiser’s return on investment. These metrics that are measureable in A/B tests in themselves are only a proxy for more long­term goals: satisfying users, increasing users, satisfying partners, and profit, which even then you could consider proxies for having a useful, high quality product and a thriving company five years from now.

**The only easy launch decisions are when all metrics get better (or at least do not get worse).** If the team has a choice between a sophisticated machine learning algorithm, and a
simple heuristic, if the simple heuristic does a better job on all these metrics, it should choose the heuristic. Moreover, there is no explicit ranking of all possible metric values. Specifically, consider the following two scenarios:

| Experiment | Daily Active Users | Revenue/Day |
|------------|--------------------|-------------|
| A          | 1 million          | $4 million  |
| B          | 2 million          | $2 million  |

If the current system is A, then the team would be unlikely to switch to B. If the current system is B, then the team would be unlikely to switch to A. This seems in conflict with rational behavior: however, predictions of changing metrics may or may not pan out, and thus there is a large risk involved with either change. Each metric covers some risk with which the team is concerned. Moreover, no metric covers the team’s ultimate concern, “where is my product going to be five
years from now”?

**Individuals, on the other hand, tend to favor one objective that they can directly optimize**. Most machine learning tools favor such an environment. An engineer banging out new features
can get a steady stream of launches in such an environment. There is a type of machine learning, multi­-objective learning, which starts to address this problem. For instance, one can
formulate a constraint satisfaction problem that has lower bounds on each metric, and optimizes some linear combination of metrics. However, even then, not all metrics are easily framed as machine learning objectives: if a document is clicked on or an app is installed, it is because that the content was shown. But it is far harder to figure out why a user visits your site. How to predict the future success of a site as a whole is [AI­complete](https://en.wikipedia.org/wiki/AI-complete), as hard as computer vision or
natural language processing.

#### Rule 40 - Keep ensembles simple.

Unified models that take in raw features and directly rank content are the easiest models to debug and understand. However, an ensemble of models (a “model” which combines the scores of other models) can work better. **To keep things simple, each model should either be an ensemble only taking the input of other models, or a base model taking many features,
but not both.** If you have models on top of other models that are trained separately, then combining them can result in bad behavior.

Use a simple model for ensembling that takes only the output of your “base” models as inputs.
You also want to enforce properties on these ensemble models. For example, an increase in the score produced by a base model should not decrease the score of the ensemble. Also, it is best
if the incoming models are semantically interpretable (for example, calibrated) so that changes of the underlying models do not confuse the ensemble model. **Also, enforce that an increase in the predicted probability of an underlying classifier does not decrease the predicted
probability of the ensemble.**

#### Rule 41 - When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.

You’ve added some demographic information about the user. You've added some information about the words in the document. You have gone through template exploration, and tuned the
regularization. You haven’t seen a launch with more than a 1% improvement in your key metrics in a few quarters. Now what?
It is time to start building the infrastructure for radically different features, such as the history of documents that this user has accessed in the last day, week, or year, or data from a different property. Use [wikidata](https://en.wikipedia.org/wiki/Wikidata) entities or something internal to your company (such as Google’s [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_Graph)). Use deep learning. Start to adjust your expectations on how much return you expect on investment, and expand your efforts accordingly. As in any engineering project, you have to weigh the benefit of adding new features against the cost of increased complexity.

#### Rule 42 - Don't expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.

Diversity in a set of content can mean many things, with the diversity of the source of the content being one of the most common. Personalization implies each user gets their own results. Relevance implies that the results for a particular query are more appropriate for that query than any other. Thus all three of these properties are defined as being different from the ordinary.
The problem is that the ordinary tends to be hard to beat.

Note that if your system is measuring clicks, time spent, watches, +1s, reshares, et cetera, you are measuring the popularity of the content. Teams sometimes try to learn a personal model with diversity. To personalize, they add features that would allow the system to personalize (some features representing the user’s interest) or diversify (features indicating if this document has any features in common with other documents returned, such as author or content), and find that those features get less weight (or sometimes a different sign) than they expect. This doesn’t mean that diversity, personalization, or relevance aren’t valuable.\* As pointed out in the previous rule, you can do post-­processing to increase diversity or relevance. If you see longer term objectives increase, then you can declare that diversity/relevance is valuable, aside from popularity. You can then either continue to use your post­-processing, or directly modify the objective based upon diversity or relevance.

<sup>[Google Research Blog - App Discovery With Google Play](https://research.googleblog.com/2016/12/app-discovery-with-google-play-part-2.html?m=1)

#### Rule 43 - Your friends tend to be the same across different products. Your interests tend not to be.

Teams at Google have gotten a lot of traction from taking a model predicting the closeness of a connection in one product, and having it work well on another. Your friends are who they are. On the other hand, I have watched several teams struggle with personalization features across product divides. Yes, it seems like it should work. For now, it doesn’t seem like it does. What has sometimes worked is using raw data from one property to predict behavior on another. Also, keep in mind that even knowing that a user has a history on another property can help. For instance, the presence of user activity on two products may be indicative in and of itself.
