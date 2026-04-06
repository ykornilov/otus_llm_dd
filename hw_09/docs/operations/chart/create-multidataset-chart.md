---
title: "Как создать мультидатасетный чарт"
description: "Следуя данной инструкции, вы сможете создать мультидатасетный чарт."
---

# Создание мультидатасетного чарта

{% note info %}

В мультидатасетных чартах можно использовать только связанные измерения (которые присутствуют во всех датасетах).

{% endnote %}

Чтобы создать [мультидатасетный чарт](../../concepts/chart/index.md#multi-dataset-charts):

{% if audience == "external" and product == "yandex-cloud" %}

{% include [datalens-workbooks-collections-note](../../../_includes/datalens/operations/datalens-workbooks-collections-note.md) %}

{% endif %}

{% if audience == "external" %}

1. Перейдите на [главную страницу]({{ link-datalens-main }}) {{ datalens-short-name }}.
1. Нажмите кнопку **Создать чарт**.

{% endif %}

{% if audience == "internal" %}

1. Перейдите на [главную страницу]({{ link-datalens-main }}) {{ datalens-short-name }}.
1. На рабочей панели слева нажмите ![image](../../../_assets/console-icons/chart-column.svg).
1. В открывшемся окне нажмите кнопку **Cоздать чарт** -> **Чарт в Wizard**.

{% endif %}

1. В разделе **Датасет** выберите датасет для визуализации. Если у вас нет датасета, [создайте его](../../dataset/create-dataset.md#create).
1. Добавьте еще один датасет. Для этого под списком датасетов нажмите **Добавить датасет** и выберите датасет для визуализации.
1. При добавлении датасета автоматически устанавливается связь по первому совпадению имени полей и типа данных полей. В открывшемся окне настройки связи между датасетами нажмите **Сохранить**.
1. При необходимости повторите пункты 3-4.
1. Выберите тип чарта, например: **Линейная диаграмма**.
1. Перетащите одно из связанных измерений в секцию **X**. Значения отобразятся в нижней части графика по оси X.
1. Перетащите показатели из разных датасетов в секцию **Y**. Значения отобразятся в виде нескольких графиков по оси Y.
1. Перетащите измерение или показатель из датасета в секцию **Фильтры**. Поле может быть пустым, тогда фильтрация не будет применена.

{% if audience == "draft" %}

{% if product == "yandex-cloud" %}

{% if audience == "external" %}

## Пример создания мультидатасетного чарта {#chart-example}

{% include [datalens-multidataset-chart-example_presets](../../../_includes/datalens/datalens-multidataset-chart-example_presets.md) %}

{% endif %}

## Пример создания мультидатасетного чарта с SQL-запросами {#sql-example}

{% if audience == "internal" %}

Для выполнения примера используйте [подключение]({{ link-datalens-main }}/connections/0sa8m0z2vqgqz-sample-ch).

{% endif %}

{% include [datalens-multidataset-chart-example](../../../_includes/datalens/datalens-multidataset-chart-example.md) %}

{% endif %}

{% endif %}

{% include [clickhouse-disclaimer](../../../_includes/clickhouse-disclaimer.md) %}