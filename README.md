## Лабораторная работа № 2 "Изучение контейнеризации, аутентификации и авторизации сервисов"

### Дисциплина: "Облачные вычислительные системы"

#### Цели и задачи работы:

1. Познакомиться со способами аутентификации и авторизации сервисов в 
   облачных системах.
2. Изучить принципы работы сервиса аутентификации и авторизации `Keycloak`.
3. Изучить особенности контейнеризации сервисов с использованием `Docker`.
4. Доработать код сервиса инференса из первой лабораторной работы для 
   реализации его аутентификации через `OAuth2` с помощью системы `Keycloak`.
5. Контейнеризовать доработанный сервис с использованием `Docker`, реализовать 
   оркестрацию используемых в работе сервисов с помощью `docker-compose`
6. Настроить автоматическую публикацию образов сервисов в репозиторий 
   `DockerHub` с помощью `Github Actions`.

#### Теоретические сведения

Под **контейнеризацией** понимают процесс развертывания программного 
обеспечения, который объединяет код приложения со всеми файлами и 
библиотеками, необходимыми для запуска в любой инфраструктуре. К 
преимуществам контейнеризации относят:
* Использование ПО в различных средах без его изменения.
* Отсутствие влияние процессов в одном контейнере на процессы в другом.
* Возможность масштабирования ПО посредством порождения новых экземпляров 
  контейнеров.
* Возможность изменять приложение, не изменяяя хостовую систему.

Одной из популярных сред выполнения контейнеров с открытым исходным кодом 
является `Docker`. Контейнеры в `Docker` – это автономные пакеты приложений и 
связанных файлов, созданных с помощью платформы `Docker`.

**Оркестрация контейнеров** – это программная технология, позволяющая 
автоматически управлять контейнерами. Это необходимо для разработки 
современных облачных приложений, поскольку приложение может содержать тысячи 
микросервисов в соответствующих контейнерах, которые необходимо в правильном 
порядке и правильным способом запустить и настроить.

Современным способом авторизации и аутентификации контейнеров в 
микроконтейнерной архитектуре является применение протокола `OAuth 2.0`. Это 
протокол авторизации, позволяющий выдать одному сервису права на доступ к
ресурсам пользователя на другом сервисе. `OAuth 2.0` определяет четыре роли:
`Владелец ресурса`, `Клиент`, `Сервер ресурсов`, `Авторизационный сервер`. 
Результатом авторизации является получение клиентом `access token` от 
авторизационного сервиса, предъявление которого серверу ресурсов необходимо 
для получения доступа.

Широко используемым в настоящее время `Identity`-провайдером является 
`Keycloak`. Keycloak основан на наборе административных пользовательских 
интерфейсов и RESTful API. Он предоставляет необходимые средства для создания 
разрешений доступа к защищенным ресурсам, связывания этих разрешений с 
политиками авторизации, применения авторизационных решений в приложениях и 
сервисах.

#### Порядок выполнения работы

1. Создайте папку на компьютере для проекта и склонируйте в нее содержимое 
   репозитория:

> `git clone https://github.com/kpdvstu/CloudCS-Lab2.git`

2. Изучите реализованный в проекте способ определения наличия прав клиента на 
   выполнение операции инференса.
3. Изучите файл `docker-compose.yml`. Разберитесь, каким образом будут 
   выполняться Docker-контейнеры. Образы, собираемые в проекте, доступны в 
   [DockerHub репозитории](https://hub.docker.com/r/kpdvstu/cloud-cs/tags).
4. Запустите контейнеры `keycloak` и `postgres` с помощью нижеприведенной 
   команды. Сервис `inference` _пока запускать **не нужно**_, поскольку он 
   требует уже настроенного Keycloak!

> `docker-compose up -d postgres keycloak`

5. Дождитесь окончания процессов запуска и инициализации контейнеров. Статус 
   их работы можно посмотреть с помощью команды:

> `docker-compose logs`

6. Удостоверившись в корректном запуске сервисов `postgres` и `keycloak`, 
   передите в браузере по адресу https://localhost:8443.
7. В открывшемся окне перейдите в раздел "Administration Console" и 
   авторизуйтесь от имени администратора Keycloak (логин: `admin`, пароль 
   указан в файле `.env`). В случае успешной авторизации перед Вами 
   откроется веб-панель управления системой `Keycloak`.
8. Теперь необходимо настроить Keycloak для корректной работы с ним сервиса 
   инференса. Создайте новую область безопасности (`realm`) с именем `inference` 
   (см. [документацию](https://www.keycloak.org/docs/latest/server_admin/#proc-creating-a-realm_server_administration_guide)). 
   Настраивать область безопасности не нужно.
9. В созданной области безопасности создайте клиента (см. [документацию](https://www.keycloak.org/docs/latest/server_admin/#proc-creating-oidc-client_server_administration_guide))
   со следующими параметрами:
   * `Client ID`: **inference-client**
   * `Always Display in Console`: **On**
   * `Client authentication`: **On**
   * `Authorization`: **On**
   * `Authentication flow`: только **Service accounts roles**, остальные 
     галочки нужно снять. При включении опции авторизации **Service accounts 
     roles** параметр включится автоматически, и отключить его не получится.
   * `Web Origins`: *
   * Остальные опции следует оставить по умолчанию.

10. На панели слева выберите пункт `Clients`, выберите вновь созданного 
    клиента и перейдите на вкладку `Credentials`. Значение **Client secret** 
    на данной вкладке вместе с указанным ранее значением **Client ID** 
    необходимо указать в файле `.env` в соответствующих полях.
11. Аналогичным образом создайте еще два клиента с теми же настройками:
    * `Privileged-client`, которому будет разрешено выполнять инференс;
    * `Unprivileged-client`, которому данное действие будет запрещено.
    
    Запишите их **Client ID** и **Client secret**, они понадобятся позже при 
    выполнении запросов к сервису.
12. Теперь нужно настроить привелегированному клиенту права на доступ к 
    ресурсу инференса. На панели слева выберите пункт `Clients`, выберите 
    клиента **inference-client** и перейдите на вкладку `Authorization`. Далее, 
    перейдите на вкладку `Scopes`. Создайте новый scope с именем `doInfer`.
13. Перейдите на вкладку `Resourses`. Создайте новый ресурс, соответствующий 
    ресурсу сервиса `/predictions`, отвечающего за инференс. Параметры ресурса:
    * `Name`: **infer_endpoint**
    * `URIs`: **/predictions**
    * `Authorization scopes`: **doInfer**
    * Остальные опции следует оставить по умолчанию.
14. Перейдите на вкладку `Policies`. Создайте новую политику типа `Client` с 
    параметрами:
    * `Name`: **inference-policy**
    * `Clients`: **privileged-client**
    * `Logic`: **Positive**
15. Перейдите на вкладку `Permissions`. Создайте новое разрешение типа 
    `scope-based permission` с параметрами:
    * `Name`: **inference-permission**
    * `Resources`: **infer_endpoint**
    * `Authorization scopes`: **doInfer**
    * `Policies`: **inference-policy**
    * `Decision strategy`: **Unanimous**
    * Остальные опции следует оставить по умолчанию.
16. Теперь можно запускать сервис инференса. Выполните команду:

> `docker-compose up -d inference`

17. Убедитесь, что сервис успешно запустился. Если в процессе запуска 
    появились ошибки, исправьте их (как правило, это связано с неправильным 
    конфигурированием `Keycloak`).
18. Проверьте работоспособность сервиса, перейдя в браузере по адресу: 
    http://localhost:8000/healthcheck.
19. Получите токен доступа привилегированного сервиса c использованием `curl` 
    (или любого другого клиента, например, `telnet`, `PuTTY`, `Postman` и др.):
> `curl --insecure --request POST
https://localhost:8443/realms/inference/protocol/openid-connect/token 
--header 'Content-Type: application/x-www-form-urlencoded'
--data-urlencode 'client_id=privileged-client'
--data-urlencode 'client_secret=<privileged-client-secret>'
--data-urlencode 'grant_type=client_credentials'` 

20. Выполните запрос на инференс от имени привилегированного пользователя:
> `curl -X POST http://localhost:8000/predictions -H "Authorization: Bearer 
<access-token>" -H 'Content-Type: application/json' -d '{"cylinders": 4, 
"displacement": 113.0, "horsepower": 95.0, "weight": 2228.0, 
"acceleration": 14.0, "model_year": 71, "origin": 3}'`

21. Убедитесь в работоспособности инференса для привилегированного 
    пользователя. Аналогичным образом запросите токен для 
    непривилегированного пользователя и попробуйте выполнить инференс с ним. 
    Убедитесь в отсутствии доступа.

#### Индивидуальное задание

*Данная лабораторная работа является составной частью курсовой работы, защищаемой студентами в конце семестра.*
При выполнении лабораторной работы можно использовать **любые** языки и фреймворки, позволяющие выполнить поставленную задачу.

1. Для своего сервиса, разработанного в процессе выполнения *первой* 
   лабораторной работы, реализуйте аутентификацию и авторизацию с 
   использованием `Keycloak`. Воспользуйтесь представленным проектом как 
   образцом.
2. Выполните контейнеризацию разработанного Вами сервиса, составьте 
   соответствующий `Dockerfile`.
3. Выполните оркестрацию всех сервисов, используемых в работе, с 
   использованием инструмента `docker-compose`. Составьте соответствующий 
   `docker-compose.yml`.
4. Добейтесь работоспособности сервиса, продемонстрируйте корректную 
   обработку запросов преподавателю для привилегированного и 
   непривилегированного пользователей.

5. Создайте свой репозиторий на `GitHub` с разработанным проектом. 
   Реализуйте CI/CD-конвейер в `GitHub Actions` для тестирования сервиса, 
   сборки Docker-образов и размещения их в DockerHub. Убедитесь в его 
   работоспособности. **Не забудьте прописать корректные [GitHub Secrets](https://docs.github.com/ru/actions/security-guides/encrypted-secrets)
   для сохранения в GitHub конфиденциальных данных!** 

6. Оформите вторую главу пояснительной записки к **курсовой работе**, описав 
   в ней следующие моменты:

* Постановку задачи.
* Реализацию модулей сервиса, ответственных за выполнение аутентификации и 
  авторизации с помощью `Keycloak`.
* Процесс настройки `Keycloak` для различных категорий пользователей, 
  использующихся в работе.
* Структуру созданных `Dockerfile` и `docker-compose.yml`, обоснование 
  применения используемых в них инструкций.
* Команды сборки образов и запуска контейнеров со скринами, 
  подтверждающими успешность их выполнения.
* Тестирование работоспособности сервиса и CI/CD (содержание запросов, 
  содержание ответов, демонстрация корректной обработки сервисом различных 
  сценариев, возникающих в процессе его использования (в том числе, 
  ошибочных), скрины с результатами тестирования и их пояснением).
* Выводы по главе с анализом полученных результатов.

#### Список вопросов к отчету работы

1. Понятие виртуализации и контейнеризации, их возможности. Преимущества и 
   недостатки каждой из технологий.
2. Принцип работы технологии контейнеризации. Пространство пользователя и 
   пространство ядра. `Kernel Namespaces`, `CGroups`, `UnionFS`.
3. Система `Docker`. Архитектура системы. Функции каждого компонента 
   архитектуры и взаимодействие между ними.
4. Процесс создания образов `Docker`. `Dockerfile` и контекст создания образа. 
   Уровни образа. `UnionFS`. Влияние кэширования на процесс создания образов.
5. Основные инструкции `Dockerfile`, их функции и правила использования. 
   Способы записи инструкций.
6. Обеспечение коммуникации между контейнерами: проброс портов и подключение 
   контейнеров.
7. Сохранение данных контейнера, понятие томов (`volumes`). Управление томами 
   `Docker`.
8. Основные команды `Docker` для работы с образами и контейнерами.
9. Понятие оркестрации Docker-контейнеров. Инструменты оркестрации Docker.
10. Инструмент `docker-compose`: назначение и принцип работы. Конфигурационный 
    YAML-файл и его формат. Определение переменных окружающей среды 
    (`environment`).
11. Определение и использование сервисов (`services`), сетей (`networks`) и 
    томов (`volumes`) в `docker-compose.yml`. Типы драйверов для сетей.
12. Сборка `Docker`-образов проекта с помощью `docker-compose` с 
    использованием `Dockerfile`. Инструкция `build` в `docker-compose.yml`.
13. Использование существующих образов проекта в `docker-compose` для 
    запуска контейнера. Инструкция `image` в `docker-compose.yml`.
14. Политики перезапуска сервисов в `docker-compose`. Инструкция `restart` в 
    `docker-compose.yml`.
15. Определение зависимости между сервисами в `docker-compose`. Инструкция 
    `depends_on` в `docker-compose.yml`.
16. Основные команды `docker-compose`. Построение и удаление образов с 
    помощью `docker-compose`. Запуск, просмотр статуса и удаление контейнеров 
    с помощью `docker-compose`.
17. Понятия идентификации, аутентификации и авторизации. Многофакторная 
    аутентификация.
18. Виды `HTTP`-авторизации, их особенности.
19. Стандарт `OpenID` и протокол `OAuth 2.0`. Основные роли в `OAuth 2.0` и 
    взаимодействие между ними.
20. Типы грантов в `OAuth 2.0`. Их особенности и области применения.
21. Понятие `Identity Providers`, основные их представители. `Keycloak`.
22. Основные механизмы контроля доступа: `ABAC`, `RBAC`, `UBAC`, `CBAC`, 
    `rule-based`, `time-based`. Их особенности и паттерны применения.

#### Список литературы для подготовки к отчету

1. Сейерс, Э. Х. Docker на практике / Э. Х. Сейерс, А. Милл ; перевод с 
   английского Д. А. Беликов. — Москва : ДМК Пресс, 2020. — 516 с. — ISBN 
   978-5-97060-772-5. — Текст : электронный // Лань : 
   электронно-библиотечная система. — URL: https://e.lanbook.com/book/131719 
   (дата обращения: 28.03.2023).

2. Моуэт, Э. Использование Docker / Э. Моуэт ; научный редактор А. А. 
   Маркелов ; перевод с английского А. В. Снастина. — Москва : ДМК Пресс, 
  2017. — 354 с. — ISBN 978-5-97060-426-7. — Текст : электронный // Лань : 
  электронно-библиотечная система. — URL: https://e.lanbook.com/book/93576 
  (дата обращения: 28.03.2023).

3. Кочер, П. С. Микросервисы и контейнеры Docker : руководство / П. С. Кочер 
   ; перевод с английского А. Н. Киселева. — Москва : ДМК Пресс, 2019. — 240 
   с. — ISBN 978-5-97060-739-8. — Текст : электронный // Лань : 
   электронно-библиотечная система. — URL: https://e.lanbook.com/book/123710 
   (дата обращения: 28.03.2023).

4. Authorization Services Guide [Электронный ресурс] : документация по 
   системе Keycloak. – [2023]. –
   Режим доступа : https://www.keycloak.org/docs/latest/authorization_services/
   (дата обращения: 28.03.2023).

5. Server Administration Guide [Электронный ресурс] : документация по 
   системе Keycloak. – [2023]. –
   Режим доступа : https://www.keycloak.org/docs/latest/server_admin/
   (дата обращения: 28.03.2023).

6. OAuth 2.0 [Электронный ресурс] : документация по протоколу OAuth 2.0. – 
   [2023]. – Режим доступа : https://oauth.net/2/ (дата обращения: 28.03.2023).

6. FastAPI [Электронный ресурс] : документация по фреймворку FastAPI. – 
   [2023]. – Режим доступа : https://fastapi.tiangolo.com/ (дата обращения: 
   28.03.2023).
