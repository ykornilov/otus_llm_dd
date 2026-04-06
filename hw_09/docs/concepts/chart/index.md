---
title: "Чарт в {{ datalens-full-name }}"
description: Чарт в {{ datalens-full-name }} — это визуализация данных из датасета в виде таблицы, диаграммы или картограммы."
---

# Чарт в {{ datalens-full-name }}

_Чарт_ — это визуализация данных из датасета в виде таблицы, диаграммы или картограммы.

В {{ datalens-short-name }} несколько видов чартов:

* [Чарты в визарде](./dataset-based-charts.md) — создаются в визарде на основе данных из одного или нескольких [датасетов](../../dataset/index.md).
* [QL-чарты](./ql-charts.md) — для построения таких чартов используется прямой запрос в источник.
{% if product != "dltech" %}* [Чарты в Editor]{% if audience == "external" %}(../../charts/editor/index.md){% endif %}{% if audience == "internal" %}(../../editor/index.md){% endif %} — визуализации и селекторы создаются с помощью кода на JavaScript.{% endif %}

В {{ datalens-short-name }} поддерживаются различные типы визуализаций: диаграммы, таблицы, географические карты и другие средства. Подробнее см. в [Справочнике визуализаций](../../visualization-ref/index.md).

Ознакомьтесь с подробной информацией о чартах в разделах:

* [{#T}](./settings.md)
* [{#T}](./measure-values.md)
{% if product != "dltech" and product != "dlent" %}* [{#T}](./multidataset-chart.md){% endif %}
{% if product == "yandex-cloud" %}* [{#T}](./versioning.md){% endif %}
{% if product == "yandex-cloud" %}* [{#T}](./inspector.md){% endif %}
{% if product == "yandex-cloud" %}* [{#T}](./access-management.md){% endif %}
{% if audience == "internal" %}* [{#T}](./alerting.md){% endif %}
{% if audience == "internal" %}* [{#T}](./comment-feed-setting.md){% endif %}
* Пошаговые инструкции:

  * [{#T}](../../operations/chart/create-chart.md)
  * [{#T}](../../operations/chart/create-sql-chart.md)
  * [{#T}](../../operations/chart/replace-dataset.md)
{% if product != "dltech" and product != "dlent" %}  * [{#T}](../../operations/chart/create-multidataset-chart.md){% endif %}
{% if audience == "internal" %}  * [{#T}](../../operations/chart/create-alert.md){% endif %}
  * [{#T}](../../operations/chart/add-description.md)
  * [{#T}](../../operations/chart/add-hierarchy.md)
  * [{#T}](../../operations/chart/config-chart-navigator.md)
{% if audience == "internal" %}  * [{#T}](../../operations/chart/use-comments.md){% endif %}
{% if audience == "internal" %}  * [{#T}](../../operations/chart/export-from-monitoring.md){% endif %}
  * [{#T}](../../operations/chart/chart-null-settings.md)
{% if audience == "external" %}  * [{#T}](../../operations/chart/create-palette.md){% endif %}
  * [{#T}](../../operations/chart/add-parameters.md)
  * [{#T}](../../operations/chart/add-guid.md)
  * [{#T}](../../operations/chart/add-parameter-chart.md)