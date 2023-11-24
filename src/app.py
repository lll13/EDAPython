# CLI : shiny run
import os
from math import ceil
from typing import List
import json
import pandas
# import shinyswatch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import asyncio
import datetime
# import pycanon 
import pycanon.anonymity as anony
from shiny import App, Inputs, Outputs, Session, reactive, render, run_app, ui

from util.log import log
from util.guessEncoding import guessEncoding
from util.guessDelimiter import guessDelimiter

# set plot font, Malgun Gothic
plt.rcParams["font.family"] = "Malgun Gothic"
# sns.set(font="Nanum Gothic", rc={"axes.unicode_minus":False}, style='darkgrid')

from util.sharedStore import sharedStore

# =====================================
# Define UI
# =====================================

store = sharedStore()
SHINY_SHAREDATA = store.get("SHINY_SHAREDATA")

enable_filedialog = SHINY_SHAREDATA.get('enable_filedialog', 1)
if enable_filedialog==1 :
    input_file_stype = {"style": "display:black;"}
else :
    input_file_stype = {"style": "display:none;"}

 
LOADING_UI = False
CLICK_COUNT = 1
CLICK_COUNT2 = 1
mappingTable = {1: 'a'}
app_ui = ui.page_fluid(
    {"style": "background-color: rgba(0, 128, 255, 0.1);border:0;padding:0;margin:0;font-size: 13px;"},
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.tags.div(
                input_file_stype,
                ui.input_file("file1", "분석 파일 선택 : ", multiple=False, button_label="파일 선택...", placeholder="파일 선택 바랍니다.", accept=".csv,.txt")
            ),
            ui.navset_tab_card(
                ui.nav(
                    "그래프",
                    ui.input_select(
                        "plotColumn",
                        "분석할 칼럼 선택 : ",
                        [],
                    ),
                    ui.input_numeric(
                        "plotObs", 
                        "분할 구간수 입력 : (0<n<11)", 10, min=1, max=10),
                    ui.input_action_button("setPlot", "그래프생성",class_="btn btn-outline-primary", checkd_autocomplete='', disabled=LOADING_UI, ),
                    value="tabPlot"
                ),
                ui.nav(
                    "통계/샘플링 분석", 
                    ui.input_select(
                        "statColumn",
                        "분석할 칼럼 선택 : ",
                        [],
                    ),
                    ui.input_numeric(
                        "statObs", 
                        "샘플 추출 건수 입력 : ", 10, min=1, max=100),
                    ui.input_slider(
                        "statSlider", "샘플 추출 구간 설정: ", min=1, max=2000, value=(100, 500)
                    ),
                    ui.input_action_button("setStat", "데이터분석 및 추출",class_="btn btn-outline-primary", disabled=LOADING_UI),
                    value="tabStat",
                
                ),
                ui.nav(
                    "특이정보분석", 
                    ui.input_checkbox("chkCatAll", "전체 범주형 칼럼 대상 분석", True),
                    ui.panel_conditional(
                        "!input.chkCatAll",
                        ui.input_selectize(
                            "selCatColumn",
                            "범주형 분석 대상 칼럼을 선택 : ",
                            [],
                            multiple=True,
                        )
                    ),
                    ui.input_numeric(
                        "freqObs", 
                        "분석 빈도수 입력 : ", 10, min=1, max=100
                    ),
                    ui.input_action_button("setFreq", "빈도분석",class_="btn btn-outline-primary", disabled=LOADING_UI),
                    ui.tags.br(""),
                    ui.input_checkbox("chkNumAll", "전체 수치형 칼럼 대상 분석", True),
                    ui.panel_conditional(
                        "!input.chkNumAll",
                        ui.input_selectize(
                            "selNumColumn",
                            "수치형 분석 대상 칼럼을 선택 : ",
                            [],
                            multiple=True,
                        )
                    ),
                    ui.input_action_button("setStat2", "기초통계분석",class_="btn btn-outline-primary", disabled=LOADING_UI),
                    value="tabOutlier"),
                ui.nav(
                    "KLT 분석", 
                    ui.input_selectize(
                        "qiColumn",
                        "준식별자 칼럼을 선택 : ",
                        [],
                        multiple=True,
                    ),
                    ui.input_action_button("setQi", "준식별자 익명지표 분석", class_="btn btn-outline-primary",disabled=LOADING_UI),
                    ui.tags.br(""),
                    ui.input_selectize(
                        "saColumn",
                        "민감정보 칼럼을 선택 : ",
                        [],
                        multiple=True,
                    ),
                    ui.input_action_button("setSa", "민감정보 익명지표 분석", class_="btn btn-outline-primary",disabled=LOADING_UI),  
                    ui.tags.br(""),
                    ui.tags.label("*민감정보 익명지표 분석은 준식별자 칼럼 선택이 선행되어야 하며,", ui.tags.br(), "준식별자 칼럼을 포함할 수 없습니다."),
                    value="tabKlt"),
                id="inTabset",
            ),
        ),
        ui.panel_main(
            ui.output_ui("dyn_ui"),
        ),
    ),
    title="DoToRi EDA",
)

# =====================================
# Define Server
# =====================================
def server(input, output, session):
    
    fileName = reactive.Value(None)
    df = reactive.Value(None)
    cntRows = reactive.Value(None)
    colNameList = reactive.Value(None)
    catColNameList = reactive.Value(None)
    numColNameList = reactive.Value(None)
    resultSetFreq = reactive.Value(None)
    setFreqBaseVal = reactive.Value(None)
    
    plotPath = reactive.Value(None)
    # mappingTable = dict
    # mappingTable[input.file1()[0]] = input.file1()[0]['name']

    @reactive.Effect
    def _():
        typeTab = input.inTabset()
        # print(f"Check input data -> {typeTab}")
        
        if input.file1() is None and fileName() is None :
            jsonData = SHINY_SHAREDATA
            
            fileName2 = jsonData["file"]
            filePath = os.path.join(jsonData["dir"], jsonData["file"])
            
            if not os.path.exists(filePath) :
                return
            
            optSep = jsonData["sep"]
            optEncode = jsonData["encode"]
            
            init(filePath, optSep, optEncode)
            fileName.set(fileName2)
            
            plotPath.set(initPlotStoragePath(fileName()))
    
    @reactive.Effect
    @reactive.event(input.file1)
    def _():
        try:
            fileInfos = input.file1()
            fileinfo = fileInfos[0]
            fileName2 = fileinfo['name']
            fileSize = fileinfo['size']
            fileType = fileinfo['type']
            filePath = fileinfo['datapath']
            optSep = guessDelimiter(filePath)
            optEncode = guessEncoding(filePath)
            
            init(filePath, optSep, optEncode)
            fileName.set(fileName2)
            
            plotPath.set(initPlotStoragePath(fileName()))
        except Exception as e:
            m = ui.modal(
            "지원하지 않는 파일 형식입니다. 다시 선택해주시길 바랍니다.",
            title="Warning",
            easy_close=True,
            footer=None,
            )
            ui.modal_show(m)
            return f"예상치 못한 오류가 발생했습니다: {e}"

    @output
    @render.ui
    def dyn_ui():
        global CLICK_COUNT
        typeTab = input.inTabset()
        if typeTab == "tabPlot":
            if(input.setPlot()==CLICK_COUNT):
                CLICK_COUNT+=1
                ui.update_action_button(id="setPlot", label="그래프생성 중...")
                return  ui.tags.br(""), ui.output_text(
                            "descPlot1"
                        ),  ui.output_plot(
                                "plot1"
                            ),  ui.output_text(
                                    "descPlot2"
                                ),  ui.output_plot("plot2") 
        elif typeTab == "tabStat":
                return  ui.tags.br(""), ui.output_text_verbatim(
                            "descStat", placeholder=True
                        ),  ui.output_table("sample")
        elif typeTab == "tabOutlier":
                return  ui.tags.br(""), ui.navset_tab_card(
                            ui.nav(
                                "범주형 데이터 분석",
                                ui.output_text_verbatim(
                                    "descFreqCat", placeholder=True
                                ),  
                                ui.input_select(
                                    "freqCatColumn2",
                                    "지정된 빈도 이하 보유 칼럼 목록 : ",
                                    [],
                                ),  
                                ui.output_table(
                                    "freqData"
                                ),
                                value="tabCat"
                            ),
                            ui.nav(
                                "수치형 데이터 분석", 
                                ui.output_text_verbatim("descNumStat", placeholder=True),
                                value="tabNum",
                            ),
                            id="mainTabSet",
                        ),
        elif typeTab == "tabKlt":
            global CLICK_COUNT2
            if(input.setSa()==CLICK_COUNT2):
                CLICK_COUNT2+=1
                ui.update_action_button(id="setSa", label="민감정보 익명지표 분석 중...")
                return  ui.tags.br(""), ui.output_text_verbatim(
                            "descQi", placeholder=True
                        ),  ui.output_text_verbatim(
                                "descSa", placeholder=True
                            ),	ui.navset_tab_card(
                                    ui.nav(
                                        "QI Data",
                                        ui.output_table("qiAnony"),
                                        value="tabQi",
                                    ),
                                    ui.nav(
                                        "QI & SA Data", 
                                        ui.output_table("saAnony"),
                                        value="tabSa",
                                    ),
                                    id="mainTabSet2",
                                ),
            # else :
            #     return  ui.tags.br(""), ui.output_text_verbatim(
            #                     "descQi", placeholder=True
            #                 ),  ui.output_text_verbatim(
            #                         "descSa", placeholder=True
            #                     ),	ui.navset_tab_card(
            #                             ui.nav(
            #                                 "QI Data",
            #                                 ui.output_table("qiAnony"),
            #                                 value="tabQi",
            #                             ),
            #                             ui.nav(
            #                                 "QI & SA Data", 
            #                                 ui.output_table("saAnony"),
            #                                 value="tabSa",
            #                             ),
            #                             id="mainTabSet2",
            #                         ),
                                
    @render.ui
    @output
    @render.text
    @reactive.event(input.setPlot)
    async def descPlot1():
        tmpCntRows = cntRows()
        m = ui.modal(
            "결과 생성 중입니다...",
            footer=None,
        )
        # ui.modal_show(m)
        with ui.Progress(min=1, max=tmpCntRows) as p:
            p.set(message="Calculation in progress", detail=None)
            msg = "데이터 처리 중...."
            msg = msg+"\n완료되면 그래프가 생성됩니다."
            # for i in range(1, tmpCntRows):
            #     p.set(i, message=msg)
            #     await asyncio.sleep(0.01)
            ui.update_action_button(id="setPlot", label="그래프생성")
            # ui.modal_remove()
                
        plotStoragePath = plotPath()
        return "빈도 기준 분석 그래프 저장경로 : " + plotStoragePath
            # return "빈도 기준 분석 그래프(파일명과 칼럼이름으로 이미지 자동 저장됨) : "
    
    @output
    @render.plot
    @reactive.event(input.setPlot)
    def plot1():
        retVal = None
        tmpDf = df()
        tmpColName = input.plotColumn()
        tmpColIndex = tmpDf.columns.get_loc(tmpColName)
        tmpDf = tmpDf[[tmpColName]]
        # print("tmpDf.head() -> ", tmpDf.head())
        # print("input.plotColumn() -> ", tmpColName)
        if tmpDf[tmpColName].dtype == "object":
            tmpDf[tmpColName] = tmpDf[tmpColName].astype('category')
        
        plt.xticks(rotation=-45)
        retVal = sns.countplot(x=tmpColName, data=tmpDf)
        retVal.set_xlabel(tmpColName[:-1])
            
        # plt.barh(tmpColName, tmpDf)
        ##20230810, set path value and save plot image
                
        imgInfo = str(tmpColIndex).zfill(2)+"_"+tmpColName+".png"
        plotStoragePathImg = os.path.join(plotPath(), imgInfo)
        makePlotStoragePath(plotPath())
        
        plt.savefig(plotStoragePathImg)
        return retVal

    @output
    @render.text
    @reactive.event(input.setPlot)
    async def descPlot2():
        tmpDf = df()
        tmpColName = input.plotColumn()
        tmpDf = tmpDf[[tmpColName]]
        if tmpDf[tmpColName].dtype == "object":
            tmpDf[tmpColName] = tmpDf[tmpColName].astype('category')

        if tmpDf[tmpColName].dtype != "category":
            return "수치형 구간 나눔 분석 그래프 : "
        else:
            return
    
    @output
    @render.plot
    @reactive.event(input.setPlot)
    def plot2():
        tmpDf = df()
        tmpColName = input.plotColumn()
        tmpDf = tmpDf[[tmpColName]]
        if tmpDf[tmpColName].dtype == "object":
            tmpDf[tmpColName] = tmpDf[tmpColName].astype('category')

        if tmpDf[tmpColName].dtype != "category":
            if input.plotObs() is None:
                obs_value = 10
            else:
                if input.plotObs()>10:
                    obs_value = 10
                elif input.plotObs()<1:
                    obs_value = 1
                else :
                    obs_value = input.plotObs()
            ax = sns.histplot(x=tmpColName, data=tmpDf, bins=obs_value)
            ax.set_xlabel(tmpColName[:-1])
            return ax
            # return sns.histplot(x=tmpColName, data=tmpDf, bins=obs_value)
        else:
            return
    
    @output
    @render.text
    @reactive.event(input.setStat)
    def descStat():
        tmpDf = df().copy()
        tmpColName = input.statColumn()
        tmpDf = tmpDf[[tmpColName]]
        tmpDf.columns = ["v1"]
        if tmpDf["v1"].dtype == "object":
            tmpDf["v1"] = tmpDf["v1"].astype('category')

        if tmpDf["v1"].dtype == "category":
            freq = tmpDf["v1"].value_counts()
            freq = freq.sort_values(ascending=True)
            # print(f"freq -> {freq}")
            # naList = freq.loc[freq.values <= 10].index.to_list()
            cnt = 0
            tmpDesc = "최대 5개까지 범주 데이터를 표시(빈도가 낮은 순으로 표시)"
            for index, value in enumerate(freq):
                # print(index, value)
                # print(freq.index[index], value, str(value))
                cnt = cnt + 1
                if cnt < 6:
                    tmpDesc = tmpDesc + "\n" + str(cnt) + ". " + freq.index[index] + " - " + str(value) + "건"
                else:
                    break
            return tmpDesc
        else:
            # check max, min and freq
            vMax = tmpDf["v1"].max()
            vMin = tmpDf["v1"].min()
            # print("vmax -> ", vMax)
            # aaa = tmpDf[tmpDf["v1"] >= vMax]
            # print("aaa -> ", aaa)
            # print("count aaa -> ", len(aaa))
            tmpDesc = "1. 평균 : " + str(tmpDf["v1"].mean())
            tmpDesc = tmpDesc + "\n" + "2. 중앙값 : " + str(tmpDf["v1"].median())
            tmpDesc = tmpDesc + "\n" + "3. 최댓값 : " + str(vMax) + " (총 " + str(len(tmpDf[tmpDf["v1"] >= vMax])) + "건)"
            tmpDesc = tmpDesc + "\n" + "4. 최솟값 : " + str(vMin) + " (총 " + str(len(tmpDf[tmpDf["v1"] <= vMin])) + "건)"
            tmpDesc = tmpDesc + "\n" + "5. 1사분위 : " + str(tmpDf["v1"].quantile(q=0.25))
            tmpDesc = tmpDesc + "\n" + "6. 3사분위 : " + str(tmpDf["v1"].quantile(q=0.75))
            return tmpDesc
    
    @output
    @render.table
    @reactive.event(input.setStat)
    def sample():
        # define dataframe
        tmpDf = df().copy()
        tmpColName = input.statColumn()
        tmpDf = tmpDf[[tmpColName]]
        # tmpDf.columns = ["v1"]

        # check index range for slice
        sliderRng = list(input.statSlider()) 
        idxStart = sliderRng[0]-1
        idxEnd = sliderRng[1]
        # print(f"start -> {idxStart}, end -> {idxEnd}")

        # check show data and slice
        tmpDf["NO"] = tmpDf.index+1 
        tmpDf = tmpDf.iloc[idxStart:idxEnd,:]

        # sampling
        # print(input.statObs())
        # tmpDf = tmpDf.sample(input.statObs(), replace=False, axis=0)
        tmpDf = tmpDf.sample(input.statObs(), replace=True)
        tmpDf = tmpDf[["NO", tmpColName]].sort_index(ascending=True)
        # print(f"tmpDF -> {tmpDf.head()}")
        return tmpDf
    
    @reactive.Effect
    @reactive.event(input.setFreq)
    def _():
        # set dataframe
        tmpDf = df()
        if input.chkCatAll():
            tmpColNameList = catColNameList()
        else:
            tmpColNameList = list(input.selCatColumn())

        # check freq base value, setFreqBaseVal
        if input.freqObs() is None:
            obs_value = 10
        else:
            if input.freqObs()>10:
                obs_value = 10
            elif input.freqObs()<1:
                obs_value = 1
            else :
                obs_value = input.freqObs()
        setFreqBaseVal.set(obs_value)
        tmpBaseVal = setFreqBaseVal()
        
        # check freq and update select list
        aColNameList = []
        aColNameListDesc = []
        for i in range(0, len(tmpColNameList)):
            freq = tmpDf[tmpColNameList[i]].value_counts()
            freq = freq.sort_values(ascending=True)
            tmpList = freq.loc[freq.values <= tmpBaseVal].index.to_list()
            if len(tmpList) >= 1:
                aColNameList.append(tmpColNameList[i])
                aColNameListDesc.append(tmpColNameList[i] + " | " + str(len(tmpList)) + "건")
        
        tmpDesc = "범주형 빈도 분석 결과"
        if len(aColNameList) >= 1:
            tmpDesc = tmpDesc + "\n기준 빈도 이하 보유 칼럼 수 : " + str(len(aColNameList))
            tmpDesc = tmpDesc + "\n아래 칼럼 목록에서 선택하면 해당 정보를 확인할 수 있습니다."
            ui.update_select(
                "freqCatColumn2",
                choices=aColNameListDesc,
                # selected=aColNameListDesc[0]
            )
        else:
            tmpDesc = tmpDesc + "기준 빈도 이하 보유 칼럼이 존재하지 않습니다."
            ui.update_select(
                "freqCatColumn2",
                choices=[],
                # selected=aColNameListDesc[0]
            )
        resultSetFreq.set(tmpDesc)

        ui.update_navs("mainTabSet", selected="tabCat")

    @output
    @render.text
    @reactive.event(input.setFreq)
    def descFreqCat():
        tmpDesc = resultSetFreq()
        return tmpDesc
    
    @output
    @render.table
    @reactive.event(input.freqCatColumn2)
    def freqData():
        # check column name
        tmpValue = input.freqCatColumn2()
        tmpValue = tmpValue.split(" | ")
        
        # check freq base value
        tmpBaseVal = setFreqBaseVal()
        # print(f"freq base value -> {tmpBaseVal}")

        # set dataframe
        tmpDf = df()
        freq = tmpDf[tmpValue[0]].value_counts()
        freq = freq.sort_values(ascending=True)
        freq = freq.loc[freq.values <= tmpBaseVal]
        freq = freq.reset_index()
        freq["NO"] = freq.index+1
        freq.columns = [tmpValue[0],"빈도","NO"]
        freq = freq[["NO",tmpValue[0],"빈도"]]
        
        return freq

    @output
    @render.text
    @reactive.event(input.setStat2)
    def descNumStat():
        # check Data and numeric columns
        tmpDf = df()
        if input.chkNumAll():
            tmpColNameList = numColNameList()
        else:
            tmpColNameList = list(input.selNumColumn())

        retDesc = None
        for i in range(0, len(tmpColNameList)):
            # define column name
            tmpColName = tmpColNameList[i]

            # check max, min and freq
            vMax = tmpDf[tmpColName].max()
            vMin = tmpDf[tmpColName].min()
            tmpDesc = str(i+1) + ". 칼럼 " + tmpColName + " 기초통계데이터"
            tmpDesc = tmpDesc + "\n    - 평균 : " + str(tmpDf[tmpColName].mean())
            tmpDesc = tmpDesc + "\n    - 중앙값 : " + str(tmpDf[tmpColName].median())
            tmpDesc = tmpDesc + "\n    - 최댓값 : " + str(vMax) + " (총 " + str(len(tmpDf[tmpDf[tmpColName] >= vMax])) + "건)"
            tmpDesc = tmpDesc + "\n    - 최솟값 : " + str(vMin) + " (총 " + str(len(tmpDf[tmpDf[tmpColName] <= vMin])) + "건)"
            tmpDesc = tmpDesc + "\n    - 1사분위 : " + str(tmpDf[tmpColName].quantile(q=0.25))
            tmpDesc = tmpDesc + "\n    - 3사분위 : " + str(tmpDf[tmpColName].quantile(q=0.75))
            if retDesc is None:
                retDesc = tmpDesc
            else:
                retDesc = retDesc + "\n\n" + tmpDesc
        
        return retDesc
    
    @reactive.Effect
    @reactive.event(input.setStat2)
    def _():
        ui.update_navs("mainTabSet", selected="tabNum")

    @output
    @render.text
    @reactive.event(input.setQi)
    def descQi():
        # check quasi-id columns
        tmpQiColumn = input.qiColumn()
        if not tmpQiColumn:
            return
        
        # check Data-Set
        tmpDf = df()
        
        # define return info
        kAnony = anony.k_anonymity(tmpDf, tmpQiColumn)
        retDesc = "1. 준식별자 대상 익명성 지표 분석"
        retDesc = retDesc + "\n  - 열정보 : " +  json.dumps(tmpQiColumn, ensure_ascii=False)
        retDesc = retDesc + "\n  - k_anonymity : " + str(kAnony)
        return retDesc

    @output
    @render.table
    @reactive.event(input.setQi)
    def qiAnony():
        # check column name
        tmpQiColumn = input.qiColumn()
        if not tmpQiColumn:
            return
        
        tmpQiColumn = list(tmpQiColumn)
        # set dataframe
        tmpDf = df()
        freq = tmpDf[tmpQiColumn].value_counts(ascending=True)
        # freq = freq.sort_values(ascending=True)
        freq = freq.reset_index()
        freq["NO"] = freq.index+1
        tmpQiColumn.extend(["빈도","NO"])
        # tmpQiColumn.append("NO")
        freq.columns = tmpQiColumn
        tmpQiColumn = tmpQiColumn[-1:]+tmpQiColumn[:-1]
        freq = freq[tmpQiColumn]
        return freq

    @reactive.Effect
    @reactive.event(input.setQi)
    def _():
        ui.update_navs("mainTabSet2", selected="tabQi")

    # @output
    # @render.text
    @reactive.Effect
    @reactive.event(input.setSa)
    # async def descSa():
    async def _():
        tmpCntRows = cntRows()
        print(f"tmpCntRows -> {tmpCntRows}")
        m = ui.modal(
            "결과 생성 중입니다...",
            footer=None,
        )
        ui.modal_show(m)
        with ui.Progress(min=1, max=tmpCntRows) as p:
            p.set(message="Calculation in progress", detail=None)
            msg = "데이터 처리 중...."
            msg = msg+"\n완료되면 QI & SA Data 탭에 데이터가 표시됩니다."
            for i in range(1, tmpCntRows):
                p.set(i, message=msg)
            ui.modal_remove()
        # return "데이터 처리 진행 중..."
    
    @output
    @render.text
    @reactive.event(input.setSa)
    def descSa():
        # check quasi-id columns
        tmpQiColumn = input.qiColumn()
        tmpSaColumn = input.saColumn()
        if not tmpQiColumn or not tmpSaColumn:
            return
        # check Data-Set
        tmpDf = df()
        
        # define return info
        lDiv = anony.l_diversity(tmpDf, tmpQiColumn, tmpSaColumn)
        entropyL = anony.entropy_l_diversity(tmpDf, tmpQiColumn, tmpSaColumn)
        tClos = anony.t_closeness(tmpDf, tmpQiColumn, tmpSaColumn)
        retDesc = "2. 민감정보 대상 익명성 지표 분석"
        retDesc = retDesc + "\n  - 열정보 : " +  json.dumps(tmpSaColumn, ensure_ascii=False)
        retDesc = retDesc + "\n  - l_diversity : " + str(lDiv)
        retDesc = retDesc + "\n  - entropy_l_diversity : " + str(entropyL)
        retDesc = retDesc + "\n  - t_closeness : " + str(tClos)

        ui.update_action_button(id="setSa", label="민감정보 익명지표 분석")
        return retDesc
    
    @output
    @render.table
    @reactive.event(input.setSa)
    def saAnony():
        # check column name
        tmpQiColumn = input.qiColumn()
        tmpSaColumn = input.saColumn()
        if not tmpQiColumn or not tmpSaColumn:
            return
        tmpQiColumn = list(tmpQiColumn)
        tmpSaColumn = list(tmpSaColumn)
        # check Data-Set
        tmpDf = df()
        # duplication check
        if any(col in tmpQiColumn for col in tmpSaColumn):
            return None

        # generate table on k-anony
        freq = tmpDf[tmpQiColumn].value_counts(ascending=True)
        # freq = freq.sort_values(ascending=True)
        freq = freq.reset_index()
        freq["ec_No"] = freq.index+1
        tmpQiColumn.extend(["빈도","ec_No"])
        # tmpQiColumn.append("NO")
        freq.columns = tmpQiColumn
        # tmpQiColumn = tmpQiColumn[-1:]+tmpQiColumn[:-1]
        # freq = freq[tmpQiColumn]
        print(f"k column -> {tmpQiColumn}, lt column -> {tmpSaColumn}")

        # generate table on k-anony and lt-column
        tmpDf = tmpDf[tmpQiColumn[:-2]+tmpSaColumn]
        # if tmpDf.columns.duplicated().sum() > 0:
        #     return None
        newDf = pandas.merge(tmpDf, freq, on=tmpQiColumn[:-2]) # 오류처리
        newDf.sort_values("ec_No", inplace=True)
        return newDf
    
    @reactive.Effect
    @reactive.event(input.setSa)
    def _():
        ui.update_navs("mainTabSet2", selected="tabSa")


    # ============================================
    # Data init~ on file change
    # ============================================
    def init(filePath, optSep, optEncode):
        # read data-set
        tmpDf = pandas.read_csv(filePath, sep=optSep , encoding=optEncode)
        df.set(tmpDf)
        cntRows.set(len(tmpDf))
        tmpColNameList = tmpDf.columns.to_list()
        colNameList.set(tmpColNameList)
        tmpCatColNameList = []
        tmpNumColNameList = []
        for i in tmpColNameList:
            if tmpDf[i].dtype == "object" or tmpDf[i].dtype == "category":
                tmpCatColNameList.append(i)
            else:
                tmpNumColNameList.append(i)
        
        catColNameList.set(tmpCatColNameList)
        numColNameList.set(tmpNumColNameList)
        # print(f"catcol list -> {catColNameList()}, numcol list -> {numColNameList()}")

        # plot area
        ui.update_select(
            "plotColumn",
            choices=tmpColNameList,
            selected=None
        )

        # if df[colNameList[0]].dtype == "object" or df[colNameList[0]].dtype == "category":
        #     print("plot1")
            
        # stat area
        ui.update_select(
            "statColumn",
            choices=tmpColNameList,
            selected=None
        )
        ui.update_slider(
            "statSlider", value=(int(len(tmpDf)*0.25), int(len(tmpDf)*0.75)), min=1, max=len(tmpDf)
        )

        # freq area
        ui.update_selectize(
            "selCatColumn",
            choices=tmpCatColNameList,
            selected=None,
            # server=True,
        )
        ui.update_select(
            "selNumColumn",
            choices=tmpNumColNameList,
            selected=None,
            # server=True,
        )

        # klt area
        ui.update_selectize(
            "qiColumn",
            choices=tmpColNameList,
            selected=None,
            # server=True,
        )
        ui.update_selectize(
            "saColumn",
            choices=tmpColNameList,
            selected=None,
            # server=True,
        )
        
    def initPlotStoragePath(fileName) :
        
        if fileName is None :
            return ""
        
        strReturnPath = None
        now = datetime.datetime.now()
        currentDate = now.strftime("%Y%m%d")
        downloads_path = os.path.join(Path.home(), "Downloads", "DOTORI", "Plot", currentDate)
        
        lfilename = os.path.splitext(fileName)[0]
        lfilename = lfilename.replace(" ", "_")
        lfilepath = os.path.join(downloads_path, lfilename)
        if os.path.isdir(lfilepath) : 
            for i in range(1, 99) :
                lfilepathTemp = lfilepath + "_" + str(i).zfill(2)
                if not os.path.isdir(lfilepathTemp) :
                    strReturnPath = lfilepathTemp
                    break
        else :
            strReturnPath = lfilepath
        
        return strReturnPath

    def makePlotStoragePath(path) :
        if not os.path.isdir(path) :
            os.makedirs(path, exist_ok=True)
        
app = App(app_ui, server)