---
title: "Как создать чарт в {{ datalens-full-name }}"
description: "Следуя данной инструкции, вы сможете создать чарт в {{ datalens-full-name }}." 
---

# Создание чарта в {{ datalens-full-name }}

## Создать чарт {#create-chart}

Чтобы создать чарт:

{% if audience == "external" and product == "yandex-cloud" %}

{% include [datalens-workbooks-collections-note](../../../_includes/datalens/operations/datalens-workbooks-collections-note.md) %}

{% endif %}

{% if product == "dltech" or product == "dlent" %}

1. Перейдите на главную страницу {{ datalens-short-name }}.
1. Нажмите кнопку **Создать чарт**.

{% endif %}

{% if product == "yandex-cloud" %}

{% if audience != "internal" %}

1. Перейдите на [главную страницу]({{ link-datalens-main }}) {{ datalens-short-name }}.
1. Нажмите кнопку **Создать чарт**.

{% else %}

1. Перейдите на [главную страницу]({{ link-datalens-main }}) {{ datalens-short-name }}.
1. На рабочей панели слева нажмите ![image](../../../_assets/console-icons/chart-column.svg) и в открывшемся окне нажмите кнопку **Cоздать чарт** -> **Чарт в Wizard**.

{% endif %}

{% endif %}

1. На верхней панели выберите [тип визуализации](../../visualization-ref/index.md).
1. Перетащите требуемые для визуализации поля данных в секции чарта.
1. Задайте [настройки чарта](../../concepts/chart/settings.md).
1. Сохраните чарт:

   1. В правом верхнем углу нажмите кнопку **Сохранить**.
   1. В открывшемся окне введите название чарта и нажмите кнопку **Сохранить**.

## Создать копию или черновик чарта {#create-chart-copy}

Чтобы создать копию чарта, воспользуйтесь одним из способов:

* В правом верхнем углу чарта нажмите значок ![image](../../../_assets/console-icons/chevron-down.svg) → **Сохранить как копию**. В открывшемся окне введите название нового чарта и нажмите **Сохранить**.
* Вверху чарта нажмите значок ![image](../../../_assets/console-icons/ellipsis.svg) → ![image](../../../_assets/console-icons/copy.svg) **Дублировать**. В открывшемся окне введите название нового чарта и нажмите **Применить**.
{% if audience == "external" %}* Перейдите на страницу воркбука, в котором расположен чарт. В строке с нужным чартом нажмите значок ![image](../../../_assets/console-icons/ellipsis.svg) → ![image](../../../_assets/console-icons/copy.svg) **Дублировать**. В открывшемся окне введите название нового чарта и нажмите **Применить**.{% endif %}

Также вы можете создать черновик чарта — в правом верхнем углу чарта нажмите значок ![image](../../../_assets/console-icons/chevron-down.svg) → **Сохранить как черновик**. Подробнее см. в разделе [{#T}](../../concepts/chart/versioning.md).

{% if product == "yandex-cloud" %}

## Примеры использования {#examples}

* [{#T}](../../tutorials/data-from-ch-to-sql-chart.md)

{% endif %}

#### См. также {#see-also}

* [{#T}](../../concepts/chart/index.md)
* [{#T}](../../concepts/chart/settings.md)
